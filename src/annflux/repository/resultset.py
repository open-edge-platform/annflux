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

import matplotlib
import pandas

from annflux.repository.dataset import Dataset
from annflux.repository.model import Model
from annflux.repository.repository import Repository, RepositoryObject
from annflux.tools.io import create_directory, split_template

try:
    import _tkinter  # noqa
except ImportError:
    matplotlib.use("Agg")


class Resultset(RepositoryObject):
    label = "resultset"
    analysis_version = "1.0"

    def __init__(self, path_or_entry):
        """
        If path_or_entry is a path then path should point to a results directory
        :param path_or_entry:
        """
        if isinstance(path_or_entry, str):
            self.path = path_or_entry
            self.source_path = self.path
        else:
            self.entry = path_or_entry
            self.path = self.entry.path

            source_path_path = os.path.join(self.path, "source_path.txt")
            self.source_path = open(source_path_path).read()

        self.predictions_path = os.path.join(self.path, "results.csv")
        self.full_logits_path = os.path.join(self.path, "full_logits.npz")
        self.last_full_path = os.path.join(self.path, "last_full.npz")
        self.last_to_logits_weights_path = os.path.join(
            self.path, "last_to_logits_weights.npz"
        )
        self.record_predictions_path = os.path.join(self.path, "results_combined.csv")

        self.analysis_directory = create_directory(
            os.path.join(self.path, "analysis-{}".format(self.analysis_version))
        )
        self.combined_predictions_path = os.path.join(
            self.analysis_directory, "predictions_plus.csv"
        )

        super(Resultset, self).__init__(
            os.path.join(self.path, "name_value_cache.json")
        )

    def get_uid(self):
        """
        Gets an UID for the Resultset based on its contents
        :return:
        """
        predictions = pandas.read_csv(self.predictions_path)

        filenames = map(str, predictions.uid)
        predicted_labels = predictions.label_predicted
        scores = map(str, predictions.score_predicted)

        return hashlib.sha224(
            (
                ",".join(filenames) + ",".join(predicted_labels) + ",".join(scores)
            ).encode("utf-8")
        ).hexdigest()

    def store_contents(self, directory, mode):
        split_index = 0
        predictions_path = split_template(self.predictions_path)
        while os.path.exists(predictions_path.format(split_index=split_index)):
            shutil.copy(predictions_path.format(split_index=split_index), directory)
            # shutil.copy(split_template(self.full_logits_path).format(split_index=split_index), directory)
            shutil.copy(
                split_template(self.last_full_path).format(split_index=split_index),
                directory,
            )
            split_index += 1
        if os.path.exists(self.last_to_logits_weights_path):
            shutil.copy(self.last_to_logits_weights_path, directory)
        shutil.copy(self.predictions_path, directory)
        shutil.copy(self.last_full_path, directory)

        with open(os.path.join(directory, "source_path.txt"), "w") as f:
            f.write(self.source_path)

    @property
    def dataset(self) -> Dataset:
        return Repository.get_ancestors(self.entry, "dataset", Dataset)

    @property
    def model(self) -> Model:
        return Repository.get_ancestors(self.entry, "model", Model)

    @property
    def resultset(self):
        result = Repository.get_ancestors(self.entry, "resultset")
        return Resultset(result) if result is not None else None

    @property
    def size(self):
        return self.cache(
            "property_size", lambda: len(pandas.read_csv(self.predictions_path))
        )

    def get_cached_stats_value(
        self, name: str, type_conversion=str, custom_specifier=None
    ):
        """
        Gets a value from the statistics cache
        :param name: name of the value
        :param type_conversion: converts value using this function
        :param custom_specifier: if given retrieves value for this specifier
        :return: value if found in cache, else "N/A
        """
        if custom_specifier is not None:
            self.set_analysis_directory()
            print("self.analysisDirectory", self.analysis_directory)
        stats_path = os.path.join(self.analysis_directory, "basic_stats.csv")
        if not os.path.exists(stats_path):
            return "N/A"
        table = pandas.read_csv(stats_path)
        names = table.name

        if name in names:
            result = type_conversion(table[names.index(name)][1])
        else:
            result = "N/A"

        self.set_analysis_directory()

        return result

    def update_basic_stats(self, name, value):
        """
        Writes name, value pairs to a basic_stats file
        :param name:
        :param value:
        :return:
        """
        basic_stats_path = os.path.join(
            create_directory(self.analysis_directory), "basic_stats.csv"
        )

        headers = ("name", "value")
        if not os.path.exists(basic_stats_path):
            table = pandas.DataFrame(data=[], columns=headers)
            names = []
        else:
            table = pandas.read_csv(basic_stats_path)
            names = table.name

        if name not in names:
            table = pandas.concat(
                [table, pandas.DataFrame(data=(name, value), columns=headers)]
            )
        else:
            table.at[table[table.name == name].index, "value"] = value

        table.to_csv(basic_stats_path)

    def set_analysis_directory(self):
        self.analysis_directory = create_directory(
            os.path.join(self.path, f"analysis-{self.analysis_version}")
        )

    def get_path_for(self, filename: str):
        """
        Returns a path for `filename` in the Resultset
        """
        return os.path.join(self.path, filename)
