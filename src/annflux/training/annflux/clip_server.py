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
import datetime
import logging
import os
import sys
import tempfile
from typing import Optional

import flask
import pandas
import torch
from flask import abort, make_response
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from clip_shared import test, Image_dataset
import numpy as np


class ClipServer(object):
    def __init__(self, folder):
        self.folder = folder
        self.labels_path = os.path.join(self.folder, "labels.csv")
        self.adapter_folder = os.path.join(folder, "adapter")
        self.load_model()

    def load_model(self):
        self.device = "cuda:0"
        torch_dtype = torch.float16
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            device_map=self.device,
            torch_dtype=torch_dtype,
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.model.load_adapter(self.adapter_folder)

    def predict(self, file_path):
        self.model.eval()
        unique_labels = pandas.read_csv(self.labels_path).class_name.values.tolist()

        def test_batch_builder(samples):
            img, caption = zip(*samples)

            inputs_ = self.processor(
                text=unique_labels,
                images=img,
                return_tensors="pt",
                padding=True,
            )
            inputs_["caption"] = np.array(caption, dtype=object)
            return inputs_

        data_test = pandas.DataFrame({"filename": [file_path], "caption": ["unkown"]})
        test_set = Image_dataset(
            root_dir="Images", data_frame=data_test, processor=self.processor
        )

        test_loader = DataLoader(
            test_set,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            collate_fn=test_batch_builder,
        )

        acc, predictions, probs, hier_probs = test(
            self.model, test_loader, test_set, unique_labels
        )

        return pandas.DataFrame(
            {"filename": [file_path], "predictions": predictions, "probability": probs}
        )


EXTENSIONS = ".jpg"
logger = logging.getLogger("clip_server")
server: Optional[ClipServer] = None


app = flask.Flask(
    __name__,
    static_url_path=os.getenv("STATIC_URL", "/static"),
)


@app.route("/v1/predict", methods=["POST"])
def analyse():
    """
    Analyses an image

    Expects POST parameters:
    * image

    :return:
    """
    uploaded_files = flask.request.files.getlist("image")
    out_json = {
        "name": "https://schemas.arise-biodiversity.nl/dsi/multi-object-multi-image#single-one-prediction-per-region",
        "generated_by": {
            "datetime": datetime.datetime.now().isoformat() + "Z",
            "version": "repo:model:8d3aca47",  # TODO
            "tag": "StreetSurfaceVis",  # TODO
        },
        "media": [],
        "region_groups": [],
        "predictions": [],
    }

    # for each uploaded image perform classification
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            if uploaded_file.filename == "":
                abort_json("received_no_files", "Did not receive any files", 400)

            if not uploaded_file.filename.lower().endswith(EXTENSIONS):
                # unsupported media type
                abort_json("unsupported_media_type", "Unsupported media type", 415)

            # - write file to temporary location
            image_basename = os.path.basename(uploaded_file.filename)
            image_id = os.path.splitext(image_basename)[0]
            tmp_file = os.path.join(tmp_dir, image_basename)
            with open(tmp_file, "wb") as f:
                f.write(uploaded_file.read())

            out_json["media"].append({"id": image_id, "filename": image_basename})
            region_group_id = "region_0"
            out_json["region_groups"].append(
                {
                    "id": region_group_id,
                    "individual_id": region_group_id,
                    "regions": ["(full)"],
                }
            )

            for r, row in server.predict(tmp_file).iterrows():
                out_json["predictions"].append(
                    {
                        "region_group_id": region_group_id,
                        "classes": {
                            "type": "multiclass",
                            "items": [
                                {
                                    "name": row.predictions,
                                    "probability": row.probability,
                                }
                            ],
                        },
                    }
                )

    return out_json


def abort_json(error_code, error_message, http_status_code):
    """
    Aborts the response, returning a JSON response
    :param http_status_code: the HTTP status code to return in the response
    """
    abort(standard_json_response(error_code, error_message, http_status_code))


def standard_json_response(error_code, error_message, http_status_code):
    if error_message is None:
        error_message = " ".join(error_code.split("_")).capitalize()
    logger.info("status: {}, error message: {}".format(http_status_code, error_message))
    response = make_response(
        flask.jsonify({"error": {"code": error_code, "message": error_message}}),
        http_status_code,
    )
    return response


def m(model_folder):
    global server
    server = ClipServer(model_folder)

    app.run(
        debug=True,  # str2bool(os.getenv("APP_DEBUG", False)),
        host="0.0.0.0",
        threaded=True,
        port=int(os.getenv("PORT", "8008")),
    )


if __name__ == "__main__":
    m(sys.argv[1])
