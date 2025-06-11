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
from os import PathLike
from pathlib import Path
from typing import Union

from annflux.repository.dataset import Dataset
from annflux.repository.repository import Repository

base_data_path = os.path.expanduser(os.path.join("~", "annflux", "data"))


class AnnfluxSource(object):
    start_labels = []
    exclusivity = []
    id_column = "image_id"
    data_path_ = None
    images_path_ = None
    working_folder_ = None
    label_column_for_unseen = "label"

    def __init__(self, folder: Union[PathLike, str] = None):
        if folder is not None:
            folder = Path(folder)
            self.data_path_ = str(folder / "images.csv")
            self.images_path_ = str(folder / "images")
            self.working_folder_ = str(folder / "annflux")

    @property
    def data_path(self):
        return (
            os.path.join(base_data_path, self.data_path_)
            if base_data_path is not None
            else self.data_path_
        )

    @property
    def images_path(self):
        return (
            os.path.join(base_data_path, self.images_path_)
            if base_data_path is not None
            else self.images_path_
        )

    @property
    def working_folder(self):
        return (
            os.path.join(base_data_path, self.working_folder_)
            if base_data_path is not None
            else self.working_folder_
        )

    @property
    def labels_path(self):
        return os.path.join(self.working_folder, "labels.json")

    @property
    def exclusivity_path(self):
        return os.path.join(self.working_folder, "exclusivity.csv")

    @property
    def split_path(self):
        return os.path.join(self.working_folder, "split.json")

    @property
    def repository(self):
        return Repository(os.path.join(self.working_folder, "datarepo"))

    @property
    def dataset(self) -> Dataset:
        return self.repository.get(label=Dataset, tag="unseen").first()

