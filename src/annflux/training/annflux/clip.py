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
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import openvino as ov
import pandas
import torch
from PIL import Image
from openvino.runtime import properties
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from annflux.repository.dataset import Dataset
from annflux.tools.mixed import get_basic_logger
from annflux.training.annflux.clip_shared import test, Image_dataset
from annflux.training.annflux.feature_extractor import (
    BaseFeatureExtractor,
    PeftTrainableMixin,
    TrainParameters,
    OpenVinoMixin,
)

pandas.options.mode.copy_on_write = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_batches(data_train):
    train_img = data_train.filename.values
    train_caption = data_train.caption.values
    train_counts = Counter(train_caption.tolist())
    names, counts = zip(*train_counts.items())
    weights = np.array(counts, dtype=float)
    weights /= weights.sum()
    train_caption_to_idx = defaultdict(lambda: [])
    for i, val in enumerate(train_caption):
        train_caption_to_idx[val].append(i)
    unique_captions = list(set(train_caption))
    batch_size = len(unique_captions)  # TODO: based on number of classes
    num_batches = 1 * (len(train_img) // batch_size)
    new_captions = []
    new_images = []
    for _ in range(num_batches):
        names_for_batch = np.random.choice(
            list(names), size=batch_size, p=weights, replace=False
        )
        for caption in names_for_batch:
            index_ = np.random.choice(train_caption_to_idx[caption])
            assert caption == train_caption[index_]
            new_captions.append(train_caption[index_])
            new_images.append(train_img[index_])

    data_train = pandas.DataFrame(
        data=zip(new_images, new_captions), columns=["filename", "caption"]
    )
    return batch_size, data_train


def train_model(
    model,
    criterion,
    optimizer,
    data_train,
    val_loader,
    processor,
    custom_batch_builder,
    num_epochs=10,
    checkp_epoch=0,
    scheduler=None,
    log=True,
    plot_file=__file__ + ".log",
    device="cuda",
):
    since = time.time()
    print("len(data_train)", len(data_train))

    my_file = None
    if log:
        my_file = open(plot_file, "a")

    pbar = tqdm(range(checkp_epoch, num_epochs))
    for epoch in pbar:
        batch_size, data_train_batch = make_batches(data_train)

        train_set = Image_dataset(
            root_dir="Images", data_frame=data_train_batch, processor=processor
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_batch_builder,
            drop_last=True,
        )

        model.train()

        running_loss = 0.0
        num_batch_labels = None

        for sample in tqdm(train_loader, f"epoch {epoch}"):
            input_ids, attention_mask, pixel_values, caption = (
                sample["input_ids"],
                sample["attention_mask"],
                sample["pixel_values"],
                sample["caption"],
            )
            assert len(set(caption)) == num_batch_labels or num_batch_labels is None
            num_batch_labels = len(set(caption))
            batch_size = input_ids.size(0)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(input_ids, pixel_values, attention_mask)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                targets = torch.arange(logits_per_image.size(0)).long().to(device)

                texts_loss = criterion(logits_per_text, targets)
                images_loss = criterion(logits_per_image, targets)
                loss = (images_loss + texts_loss) / 2.0

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * batch_size

        train_loss = running_loss / len(train_loader)

        model.eval()

        running_loss = 0.0

        with torch.no_grad():
            for sample in val_loader:
                input_ids, attention_mask, pixel_values = (
                    sample["input_ids"],
                    sample["attention_mask"],
                    sample["pixel_values"],
                )
                batch_size = input_ids.size(0)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                pixel_values = pixel_values.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(input_ids, pixel_values, attention_mask)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text

                    targets = torch.arange(logits_per_image.size(0)).long().to(device)

                    texts_loss = criterion(logits_per_text, targets)
                    images_loss = criterion(logits_per_image, targets)
                    loss = (images_loss + texts_loss) / 2.0

                # statistics
                running_loss += loss.item() * batch_size

        val_loss = running_loss / len(val_loader)
        if log:
            data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(my_file, header=False, index=False)
        # print()

        pbar.set_description(
            "train loss {:.4} val loss {:.4}".format(train_loss, val_loss)
        )
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class ClipFeatureExtractor(BaseFeatureExtractor, PeftTrainableMixin, OpenVinoMixin):


    def __init__(self):
        self.huggingface_clip_name = os.getenv("HUGGINGFACE_CLIP_NAME")

        super().__init__()

    def convert_model(self):
        input_labels = [
            "cat",
            "dog",
        ]
        text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

        image = Image.new("RGB", (224, 224))

        inputs = self.processor(
            text=text_descriptions, images=[image], return_tensors="pt", padding=True
        )
        self.model.config.torchscript = True
        ov_model = ov.convert_model(self.model, example_input=dict(inputs))
        ov.save_model(
            ov_model, "clip-vit-base-patch32.xml"
        )  # TODO: align with pytorch model

        return ov_model

    def load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16
        self.model = CLIPModel.from_pretrained(
            self.huggingface_clip_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
        )
        self.processor = CLIPProcessor.from_pretrained(self.huggingface_clip_name)

    def compute_features(
        self, dataset: Dataset, multi=False, batch_size=32
    ) -> np.array:
        #

        try:
            ov_model = self.convert_model()
        except RuntimeError:
            ov_model = None

        if ov_model:
            #
            core = ov.Core()
            print(f"{core.available_devices=}")
            device = "MULTI:GPU,CPU"  # core.available_devices[0]
            # compile model for loading on device
            config = {
                # hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT,
                properties.inference_num_threads(): 8,
                properties.hint.enable_cpu_pinning(): False,
            }

            self.compiled_model = core.compile_model(ov_model, device, config)
            # obtain output tensor for getting predictions

            features = []
            for image_path_ in tqdm(
                dataset.as_dataframe().filename,
                desc="Computing CLIP features using OpenVINO",
            ):
                feature = self.compute_ov_feature(image_path_)
                features.append(feature)
            return np.vstack(features)
        else:
            features = []
            for image_path_ in tqdm(
                dataset.as_dataframe().filename, desc="Computing CLIP features"
            ):
                with torch.no_grad():
                    with torch.autocast(self.device):
                        try:
                            image = Image.open(image_path_)
                        except:  # noqa
                            print(f"Failed to read {image_path_}")
                            image = Image.new("RGB", (299, 299))
                        inputs = self.processor(
                            text=["a photo of a cat", "a photo of a dog"],
                            images=image,
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs.to(self.device)
                        outputs = self.model(**inputs)
                features.append(outputs[3].cpu().numpy())

            return np.vstack(features)

    def compute_ov_feature(self, image_path_):
        try:
            image = Image.open(image_path_)
        except:  # noqa
            print(f"Failed to read {image_path_}")
            image = Image.new("RGB", (299, 299))
        inputs = self.processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        feature = self.compiled_model(dict(inputs))[self.compiled_model.output(3)]
        return feature

    def train_peft(
        self,
        data: pandas.DataFrame,
        out_folder: os.PathLike | str,
        train_parameters: TrainParameters,
        logger=get_basic_logger("clip:train_peft"),
    ):
        """
        Assumes columns 'filename', 'label_true'
        """
        if isinstance(out_folder, str):
            out_folder = Path(out_folder)
        # TODO: exclude test
        counts = (
            data.groupby("label_true").size().to_frame(name="count").reset_index()
        )  # TODO: multilabel
        data["caption"] = data["label_true"]  # TODO: commas to spaces
        data = data[data["label_true"].isin(counts[counts["count"] >= 6]["label_true"])]
        #
        unique_labels = data.caption.unique()
        class_to_label_path = out_folder / "labels.csv"
        pandas.DataFrame(
            data={"class_name": unique_labels, "index": list(range(len(unique_labels)))}
        ).to_csv(class_to_label_path)
        #
        data_train = data[data.subset != "test"]
        data_test = data[data.subset == "test"]
        logger.info(f"{data_train.size=}")
        logger.info(f"{data_test.size=}")

        data_train, data_val = train_test_split(
            data_train,
            test_size=0.1,
            stratify=data_train["label_true"],
            random_state=42,
        )

        model = self.model

        np.random.seed(42)
        torch.manual_seed(42)

        val_set = Image_dataset(
            root_dir="Images", data_frame=data_test, processor=self.processor
        )
        test_set = Image_dataset(
            root_dir="Images", data_frame=data_test, processor=self.processor
        )
        test_true_vals = list(x[1] for x in test_set)
        unique_labels = list(data_test.caption.unique())

        def custom_batch_builder(samples):
            img, caption = zip(*samples)

            inputs_ = self.processor(
                text=caption, images=img, return_tensors="pt", padding=True
            )
            inputs_["caption"] = np.array(caption, dtype=object)
            return inputs_

        def test_batch_builder(samples):
            img, caption = zip(*samples)

            inputs_ = self.processor(
                text=unique_labels, images=img, return_tensors="pt", padding=True
            )
            inputs_["caption"] = np.array(caption, dtype=object)
            return inputs_

        train_set_size = len(data_train)
        print("Train set size:", train_set_size)
        val_set_size = len(val_set)
        print("Val set size:", val_set_size)

        criterion = nn.CrossEntropyLoss()

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        val_loader = DataLoader(
            val_set,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_batch_builder,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            collate_fn=test_batch_builder,
        )

        model = model.to(self.device)
        inputs = next(iter(val_loader))
        for key in inputs.keys():
            print("Sample {} shape ".format(key), inputs[key].shape)

        model.eval()
        test(
            model, test_loader, test_set, unique_labels, self.device
        )

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules="all-linear",  # ["q_proj", "v_proj"],  # "all-linear",
            lora_dropout=0.1,
            bias="none",
        )

        lora_model = get_peft_model(model, config)
        print_trainable_parameters(lora_model)
        #

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model = train_model(
            lora_model,
            criterion,
            optimizer,
            data_train,
            val_loader,
            self.processor,
            custom_batch_builder,
            num_epochs=train_parameters.num_epochs,
            scheduler=None,
            device=self.device,
        )
        model.save_pretrained(out_folder / "adapter")
        model.eval()
        acc, predictions, probs, hier_probs = test(
            model, test_loader, test_set, unique_labels
        )
        print("perf", acc)
        print(accuracy_score(predictions, test_true_vals))
        data_test["predictions"] = predictions
        data_test["probability"] = probs
        for level in range(6):
            data_test[f"level_{level}"] = [
                hier_prob.get(level)[0] if hier_prob.get(level) else None
                for hier_prob in hier_probs
            ]
            data_test[f"level_{level}_probability"] = [
                hier_prob.get(level)[1] if hier_prob.get(level) else None
                for hier_prob in hier_probs
            ]

        data_test.to_csv(out_folder / "predicted.csv")

        self.model.config.torchscript = True

        return class_to_label_path
