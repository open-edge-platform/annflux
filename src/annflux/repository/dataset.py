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
import hashlib
import os
import shutil
from typing import List, Dict

import pandas

from annflux.repository.repository import RepositoryObject, Repository


def label_to_name_func(
    label: str, label_to_name: Dict[str, str], allow_unknown_labels=False
):
    if label not in label_to_name and allow_unknown_labels:
        species = "n/a"
    else:
        species = label_to_name[label]
    return species.strip()


class Dataset(RepositoryObject):
    label = "dataset"

    def __init__(self, h5path_or_entry, taxon_mapping_path=None):
        if isinstance(h5path_or_entry, str):
            self.path = h5path_or_entry
            if os.path.isdir(self.path):
                self.path = os.path.join(self.path, "dataset.hdf5")
            if taxon_mapping_path is None:
                taxon_mapping_path = os.path.join(
                    os.path.dirname(self.path), "taxon_mapping.csv"
                )
            self.species_mapping_path = taxon_mapping_path
            self.source_path = os.path.dirname(h5path_or_entry)
        else:
            self.path = os.path.join(h5path_or_entry.path, "dataset.csv")
            self.species_mapping_path = os.path.join(
                h5path_or_entry.path, "taxon_mapping.csv"
            )
            self.extra_info_path = os.path.join(h5path_or_entry.path, "extra_info.csv")
            self.entry = h5path_or_entry

            super(Dataset, self).__init__(
                os.path.join(h5path_or_entry.path, "name_value_cache.json")
            )

    @property
    def uid(self):
        return self.entry.uid if hasattr(self, "entry") else "no_uid"

    def get_uid(self):
        data = pandas.read_csv(self.path)

        filenames = map(lambda x: os.path.split(x)[-1], data["filename"])
        labels = data["label"]
        labels_morph = data["label_morph"] if "label_morph" in data else None

        if labels_morph is None:
            return hashlib.sha224(
                (",".join(filenames) + ",".join(labels)).encode("utf-8")
            ).hexdigest()
        else:
            return hashlib.sha224(
                (
                    ",".join(filenames) + ",".join(labels) + ",".join(labels_morph)
                ).encode("utf-8")
            ).hexdigest()

    def store_contents(self, directory, mode):
        shutil.copy(self.path, os.path.join(directory, "dataset.csv"))
        shutil.copy(
            self.species_mapping_path, os.path.join(directory, "taxon_mapping.csv")
        )
        with open(os.path.join(directory, "source_path.txt"), "w") as f:
            f.write(self.source_path)

    def as_dataframe(self) -> pandas.DataFrame:
        """
        Returns dataset as pandas DataFrame
        has columns ["uid", "label", "taxon_name", "set", "record_id", "label_morph"]
        :return: the dataframe
        """
        return pandas.read_csv(self.path, dtype={"uid": str})

    @property
    def size(self):
        return self.cache("property_size", lambda: len(self.as_dataframe()))

    @property
    def classes(self):
        return set(
            self.cache(
                "property_classes",
                lambda: self.as_dataframe().label.tolist(),
            )
        )

    @property
    def num_classes(self):
        return self.cache("property_num_classes", lambda: len(self.classes))

    @property
    def dataset(self):
        result = Repository.get_ancestors(self.entry, "dataset")
        return Dataset(result) if result is not None else None

    @property
    def class_names(self) -> List[str]:
        """Returns class name per record"""
        return self.cache("property_class_names", self._class_names)

    def _class_names(self):
        species_map = pandas.read_csv(self.species_mapping_path, dtype=str)
        class_name_by_id = dict(list(zip(species_map.taxon_id, species_map.taxon_name)))
        return [label_to_name_func(class_, class_name_by_id) for class_ in self.classes]
