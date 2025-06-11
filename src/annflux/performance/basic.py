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
import itertools
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from annflux.tools.core import AnnFluxState

MultilabelPrediction = List[Tuple[str]]


def compute_performance(
    predicted_test: MultilabelPrediction,
    true_test: MultilabelPrediction,
    state: AnnFluxState,
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    certain_threshold=0.95,
):
    performance_graph_path = os.path.join(state.working_folder, "performance.json")
    num_train_val = len(state.labeled_indices)
    if len(true_test) > 0 and len(predicted_test) > 0:
        binarizer = MultiLabelBinarizer()
        binarizer.fit(true_test)
        acc_test = accuracy_score(
            binarizer.transform(true_test), binarizer.transform(predicted_test)
        )
        write_performance(
            performance_graph_path, acc_test, num_train_val, len(true_test)
        )
    labels = list(set(itertools.chain(*[x_.split(",") for x_ in annotations.values()])))
    detailed_performance_table = []
    for label_ in labels:
        per_image_true = np.array([int(label_ in x_) for x_ in true_test])
        per_image_predicted = np.array([int(label_ in x_) for x_ in predicted_test])

        precision, recall, fscore, support = precision_recall_fscore_support(
            per_image_true, per_image_predicted
        )
        if len(precision) > 1:
            print(
                f"{label_} precision={precision[1]:.2f} recall={recall[1]:.2f} {support[1]}"
            )
            detailed_performance_table.append(
                (label_, precision[1], recall[1], support[1])
            )
    out_table = pandas.DataFrame(
        data=detailed_performance_table,
        columns=("label", "precision", "recall", "support"),
    )

    # compute how many are certain according to a threshold
    num_certain = defaultdict(lambda: 0)
    num_uncertain = defaultdict(lambda: 0)
    num_labeled = defaultdict(lambda: 0)
    data.scores_predicted = data.scores_predicted.astype(str)
    for _, row in data.iterrows():
        if (
            row.label_predicted is not None
            and not pandas.isna(row.label_predicted)
            and not pandas.isna(row.scores_predicted)
        ):
            predicted_labels = row.label_predicted.split(",")
            predicted_probs = map(float, row.scores_predicted.split(","))
            for label_, prob_ in zip(predicted_labels, predicted_probs):
                if prob_ > certain_threshold:
                    num_certain[label_] += 1
                else:
                    num_uncertain[label_] += 1
        #
        for label_ in (
            row.label_true.split(",")
            if not pandas.isna(row.label_true) and row.label_true is not None
            else []
        ):
            num_labeled[label_] += 1

    out_table["num_predicted_certain"] = [
        num_certain.get(x_, 0) for x_ in out_table.label
    ]
    out_table["num_predicted_uncertain"] = [
        num_uncertain.get(x_, 0) for x_ in out_table.label
    ]
    out_table["num_labeled"] = [num_labeled.get(x_, 0) for x_ in out_table.label]
    out_table.to_csv(os.path.join(state.working_folder, "detailed_performance.csv"), index=False)


def write_performance(performance_graph_path, acc_test, num_train_val, num_test):
    if os.path.exists(performance_graph_path):
        j_performance = json.load(open(performance_graph_path))
    else:
        j_performance = {"test_performance": []}
    j_performance["test_performance"].append([num_train_val, num_test, acc_test])
    with open(performance_graph_path, "w") as f:
        json.dump(j_performance, f, indent=2)


def write_performance_key_val(performance_graph_path, key, val):
    if os.path.exists(performance_graph_path):
        j_performance = json.load(open(performance_graph_path))
    else:
        j_performance = {"test_performance": []}
    j_performance[key] = val
    with open(performance_graph_path, "w") as f:
        json.dump(j_performance, f, indent=2)
