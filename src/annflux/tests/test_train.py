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

import os.path
import shutil
import tempfile
import unittest
from pathlib import Path

from annflux.data.envdataset.data import EnvDataset
from annflux.shared import AnnfluxSource
from annflux.train_indeed_image import init_folder, add_annotations_and_set
from annflux.training.annflux.clip import ClipFeatureExtractor
from annflux.training.annflux.feature_extractor import TrainParameters



class TestTrain(unittest.TestCase):

    def setUp(self):
        self.data_source = EnvDataset()
        self.data_source.download()
        self.data_folder = Path(os.path.expanduser("~/annflux/data/envdataset"))
        if os.path.isdir(self.data_folder):
            shutil.rmtree(self.data_folder)
        self.data_source.copy_to(self.data_folder)
        self.annflux_folder = self.data_folder / "annflux"
        self.annflux_data_path = self.annflux_folder / "annflux.csv"

    def test_train(self):
        self.source = AnnfluxSource(self.data_folder)
        init_folder(self.source)

        clip = ClipFeatureExtractor()

        clip.load_model()

        shutil.copy(self.data_source.true_labels_path, self.source.labels_path)

        data = add_annotations_and_set(self.source.dataset, self.source)
        clip.train_peft(data, tempfile.mkdtemp(),
                        TrainParameters(num_epochs=3))
