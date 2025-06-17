# Copyright 2025 Intel Corporation
# Copyright 2025 Naturalis Biodiversity Center
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
import json
import os

import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from annflux.algorithms.embeddings import compute_tsne
from annflux.performance.basic import write_performance
from annflux.shared import AnnfluxSource
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.tools.data import color_and_label
from annflux.tools.io import read_table_pandas, numpy_load


def embed_and_prepare(source: AnnfluxSource, show=False, compute_performance=False):
    working_folder = source.working_folder
    repo = Repository(os.path.join(working_folder, "datarepo"))
    folder = repo.get(label=Resultset, tag="unseen").last().path
    labels_path = os.path.join(working_folder, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            annotations = json.load(f)
    else:
        annotations = {}
    data: pandas.DataFrame = read_table_pandas(f"{folder}/results.csv")
    data.label_predicted = data.label_predicted.astype(str)
    data.uid = data.uid.astype(str)
    extra_predictions_path = f"{folder}/predictions.csv"
    if os.path.exists(extra_predictions_path):
        extra_predictions_ = pandas.read_csv(extra_predictions_path)
        data = pandas.merge(data, extra_predictions_, on="uid")
        data["label_predicted"] = data["prediction"]
    #
    with open(os.path.join(working_folder, "split.json")) as f:
        test_uids = set(json.load(f)["test"])
    labeled_test_uids = test_uids.intersection(set(annotations.keys()))
    labeled_test_data = data[data.uid.isin(labeled_test_uids)]
    true_test = [annotations[uid_].split(",") for uid_ in labeled_test_data.uid]
    binarizer = MultiLabelBinarizer()
    binarizer.fit(true_test)
    if compute_performance:
        acc_test = accuracy_score(
            binarizer.transform(true_test),
            binarizer.transform(
                [x_.split(",") if not pandas.isna(x_) else [] for x_ in labeled_test_data.label_predicted]
            ),
        )
        write_performance(
            acc_test,
            len(annotations) - len(labeled_test_uids),
            len(true_test),
            working_folder,
        )
    npz_path = f"{folder}/custom.npz"
    npy_path = f"{folder}/custom.npy"
    if os.path.exists(npz_path):
        features = numpy_load(npz_path, "arr_0")
    elif os.path.exists(npy_path):
        features = np.load(npy_path)
    else:
        features = numpy_load(f"{folder}/last_full.npz", "lastFull")
    embedding = compute_tsne(features)
    embedding -= np.min(embedding, axis=0, keepdims=True)
    embedding /= np.max(embedding, axis=0, keepdims=True)
    embedding *= 40
    embedding -= 20
    import matplotlib.pyplot as plt
    sel = np.arange(
        len(embedding)
    )
    data = data.iloc[sel]
    data["e_0"] = embedding[sel, 0]
    data["e_1"] = embedding[sel, 1]
    data["uid"] = data["uid"].apply(lambda x_: x_)
    data["in_test"] = data["uid"].apply(lambda x_: int(x_ in test_uids))
    color_and_label(data, annotations)
    data.to_csv(os.path.join(working_folder, "annflux.csv"), index=False)
    if show:
        plt.scatter(embedding[sel, 0], embedding[sel, 1], c=data.score_predicted)
        plt.show()

