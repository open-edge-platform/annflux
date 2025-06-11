# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm



def test(
    model,
    val_loader,
    val_set,
    unique_captions,
    device="cuda",
):
    predicted_captions = []
    max_probs = []
    hier_predictions = []
    with torch.no_grad():
        running_corrects = 0.0
        i_ = 0
        for sample in tqdm(val_loader):
            input_ids, attention_mask, pixel_values, captions = (
                sample["input_ids"],
                sample["attention_mask"],
                sample["pixel_values"],
                sample["caption"],
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            # print("input_ids", input_ids)
            with torch.set_grad_enabled(False):
                outputs = model(input_ids, pixel_values, attention_mask)
                logits_per_image = outputs.logits_per_image

                _, predictions = torch.max(logits_per_image, 1)
                probs = torch.softmax(logits_per_image, 1).cpu().numpy()
                tmp__ = defaultdict(lambda: set())
                for pred_val, true_val in zip(predictions.cpu().numpy().tolist(), captions):
                    pred_caption = unique_captions[pred_val]
                    tmp__[true_val].add(pred_caption)
                    # print(pred_caption, " ", true_val)
                    running_corrects += (pred_caption == true_val) * 1
                    predicted_captions.append(pred_caption)

                # hierarchical prediction
                for b_, im_prob in enumerate(probs):
                    # print("im_prob.shape", im_prob.shape)
                    probs_per_level: Dict[int, Dict[str, float]] = {}
                    for i_, class_prob_ in enumerate(im_prob):
                        for l_, name_ in enumerate(unique_captions[i_].split()):
                            if l_ not in probs_per_level:
                                probs_per_level[l_] = defaultdict(lambda: 0)
                            probs_per_level[l_][name_] += class_prob_
                    new_probs_per_level: Dict[int, Tuple[str, float]] = {}
                    for l_, probs_ in probs_per_level.items():
                        max_name = max(probs_, key=probs_.get)
                        new_probs_per_level[l_] = (max_name, float(probs_[max_name]))
                    hier_predictions.append(new_probs_per_level)

                max_probs.extend(np.max(probs, 1).tolist())
            i_ += len(sample)

        val_acc = running_corrects / len(val_set)
        # val_loss = running_loss.item() / len(val_set)
    return val_acc, predicted_captions, max_probs, hier_predictions


class Image_dataset(object):
    def __init__(self, root_dir, data_frame, processor):  # noqa
        self.root_dir = root_dir
        self.data_frame = data_frame
        self.processor = processor

        self.data_list = []

        for _, row in self.data_frame.iterrows():
            file_name = row["filename"]
            sentence_list = [row["caption"]]

            self.data_list.append([file_name, sentence_list])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, captions = self.data_list[idx]
        try:
            img = Image.open(os.path.join(self.root_dir, image_name))
        except Exception as e:
            print(e)
            raise

        # load caption randomly
        # print(captions)
        caption = captions[0]

        return img, caption
