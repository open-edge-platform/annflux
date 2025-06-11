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
from __future__ import annotations

import os.path
import shutil
import zipfile
from io import BytesIO
from os import PathLike

import requests

from annflux.tools.mixed import get_basic_logger

logger = get_basic_logger("data")


class DataSource:
    url = None
    name = None
    hash = None  # TODO

    def __init__(self):
        self.out_folder = os.path.expanduser(f"~/annflux/datasources/{self.name}")
        print(self.out_folder)
        if not os.path.isdir(self.out_folder):
            self.download()

    def download(self):
        if self.url is None:
            return
        req = requests.get(self.url)

        zipfile_ = zipfile.ZipFile(BytesIO(req.content))
        zipfile_.extractall(self.out_folder)
        logger.warning(f"Extracted zip to {self.out_folder}")

    @property
    def folder(self):
        return self.out_folder

    def copy_to(self, folder: str | PathLike):
        print(self.out_folder)
        shutil.copytree(self.out_folder, folder)
        logger.warning(f"Copied to {self.out_folder}")


class EnvDataset(DataSource):
    url = None
    name = None
    out_folder = os.getenv("USER_DATASET_PATH")
    true_labels_path = os.path.join(out_folder, "true_labels.json")

    def __init__(self):  # noqa
        pass
