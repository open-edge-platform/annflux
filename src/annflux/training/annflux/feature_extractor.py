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
import abc
import shutil
import tempfile
from dataclasses import dataclass
from os import PathLike

import numpy as np
import pandas

from annflux.repository.dataset import Dataset
from annflux.repository.model import Model
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset


def make_resultset(dataset: Dataset, features: np.array, repo: Repository) -> Resultset:
    data = dataset.as_dataframe()
    tmp_folder = tempfile.mkdtemp()
    try:
        resultset = Resultset(tmp_folder)
        np.savez(resultset.last_full_path, lastFull=features)
        # TODO(use actual prediction when peft/deep training)
        data["label_predicted"] = "foo,bar"
        data["score_predicted"] = np.clip(
            [feature_[0] for feature_ in features], 0.0, 1.0
        )
        data.to_csv(resultset.predictions_path, index=False)
        model = repo.get(label=Model, tag="trained").first()
        repo.commit(
            resultset,
            ancestors=[dataset, model] if model is not None else [dataset],
            tag="unseen",
            allow_mixed_tags=True
        )
    except:  # noqa
        print(f"{tmp_folder=}")
        raise
    shutil.rmtree(tmp_folder)
    return resultset


@dataclass
class TrainParameters:
    num_epochs: int


class BaseFeatureExtractor(abc.ABC):
    def load_model(self):
        pass

    def compute_features(
        self, dataset: Dataset, multi=False, batch_size=32
    ) -> np.array:
        pass


class PeftTrainableMixin(abc.ABC):
    def train_peft(
        self,
        data: pandas.DataFrame,
        out_folder: PathLike | str,
        train_parameters: TrainParameters,
    ):
        pass


class OpenVinoMixin(abc.ABC):
    def convert_model(self):
        pass
