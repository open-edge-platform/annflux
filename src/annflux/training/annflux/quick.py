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
import logging
import os
import time
from collections import defaultdict
from typing import Set, Dict, List

import faiss
import numpy as np
import pandas
from tqdm import tqdm

from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.algorithms.feature_reconstruction_error import compute_fre
from annflux.algorithms.most_needed import compute_most_needed
from annflux.performance.basic import (
    compute_performance,
    write_performance_key_val,
    get_performance_key_val,
)
from annflux.tools.core import AnnFluxState
from annflux.tools.data import canon_, color_and_label
from annflux.tools.io import numpy_load
from annflux.tools.mixed import remove_sys


min_prev_near_labeled_perc = 1.0  # TODO: configurable


def quick_reclassification(
    state: AnnFluxState, knn_type="quick", logger: logging.Logger = None
):
    """
    Trains a quick new model using kNN
    """
    start_time = time.time()

    repo = Repository(os.path.join(state.working_folder, "datarepo"))
    result_set: Resultset = repo.get(label=Resultset, tag="unseen").last()
    folder = result_set.path

    data = pandas.read_csv(
        os.path.join(state.data_folder, "annflux", "annflux.csv"),
        dtype={"label_predicted": str, "score_true": float, "uid": str},
    )
    data.reset_index(drop=True, inplace=True)
    print("instant_reclassification 2", time.time() - start_time)
    with open(state.labels_path) as f:
        annotations = json.load(f)

    with open(os.path.join(state.working_folder, "split.json")) as f:
        test_uids = set(json.load(f)["test"])
    print("instant_reclassification 3", time.time())

    reload_features = state.cache_for != result_set.entry.uid
    print(
        "version_for_recompute",
        state.version_for_recompute,
        result_set.entry.uid,
        state.trained_for_version_pre,
    )
    if reload_features:
        state.g_quick_status = "Loading features"

        state.features = numpy_load(f"{folder}/last_full.npz", "lastFull")

    print(len(data), len(state.features))
    assert len(data) == len(state.features)

    annotated_uids = set(annotations.keys())
    pseudolabel_uids = {}
    use_pseudo_label = False
    if "as_pseudolabel" in data.columns and use_pseudo_label:
        pseudolabel_uids = set(data[data.as_pseudolabel == 1].uid.values.tolist())
        annotated_uids = annotated_uids.union(pseudolabel_uids)
    print("|pseudolabel_uids|", len(pseudolabel_uids))
    tmp_ = annotated_uids - test_uids
    # old_labeled_indices = copy.deepcopy(labeled_indices) if labeled_indices is not None else None
    state.labeled_indices = sorted(
        [i for i, uid in enumerate(data.uid.values) if uid in tmp_]
    )
    print("instant_reclassification 4", time.time() - start_time)
    uid_to_predicted = dict(zip(data.uid, data.label_predicted))

    #
    data["label_undetermined"] = None
    for i, uid in enumerate(data.uid.values):
        if annotations.get(uid):
            labels = annotations.get(uid).split(",")
            sure_labels = []
            for label_ in labels:
                if label_.endswith("=?"):
                    data.at[i, "label_undetermined"] = (
                        label_
                        if pandas.isna(data.at[i, "label_undetermined"])
                        else data.at[i, "label_undetermined"] + "," + label_
                    )
                else:
                    sure_labels.append(label_)
            annotations[uid] = ",".join(sorted(sure_labels))
    print(
        "set(annotations.values())",
        len(set(annotations.values())),
        set(annotations.values()),
    )
    #

    state.label_array = np.array(
        [
            (
                annotations.get(uid).split(",")
                if annotations.get(uid)
                else (
                    uid_to_predicted.get(uid).split(",")
                    if uid in pseudolabel_uids
                    else None
                )
            )
            if uid not in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ],
        dtype="object",
    )

    state.label_array = np.array(
        [remove_sys(x_) for x_ in state.label_array], dtype="object"
    )
    state.label_array_test = np.array(
        [
            (annotations.get(uid).split(",") if annotations.get(uid) else None)
            if uid in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ],
        dtype="object",
    )
    print("state.label_array_test", state.label_array_test)
    print("|label_array_test|", len(state.label_array_test))
    test_indices = set([i for i, uid in enumerate(data.uid.values) if uid in test_uids])
    print("|test_uids|", len(test_uids))
    print("|test_indices|", len(test_indices))
    state.labeled_test_indices = sorted(
        [
            i
            for i, uid in enumerate(data.uid.values)
            if uid in test_uids and uid in annotations
        ]
    )
    print("|labeled_test_indices|", len(state.labeled_test_indices))
    print("instant_reclassification 5", time.time() - start_time)

    # print("labeled_indices", labeled_indices)
    k = 110
    knn_results_path = result_set.get_path_for("knn_results.npz")
    if not os.path.exists(knn_results_path):
        state.g_quick_status = "computing knn index"
        knn_index = faiss.index_factory(
            state.features.shape[1],
            "Flat",
            {"inner": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}["l2"],
        )
        state.features *= 1 - 1e-2 * np.random.rand(
            state.features.shape[0], state.features.shape[1]
        )
        print(state.features.shape)

        knn_index.train(state.features)
        knn_index.add(state.features)

        state.all_distances, state.all_indices = knn_index.search(state.features, k=k)

        np.savez(
            knn_results_path,
            all_distances=state.all_distances,
            all_indices=state.all_indices,
        )
    else:
        logger.info(f"Loading kNN results from {knn_results_path}")
        knn_results = np.load(knn_results_path)
        state.all_distances, state.all_indices = (
            knn_results["all_distances"],
            knn_results["all_indices"],
        )

    state.cache_for = result_set.entry.uid
    state.version_for_recompute = (
        result_set.entry.uid + "_" + str(state.trained_for_version_pre)
    )

    time_start = time.time()

    if knn_type == "standard":
        knn_index_train = faiss.index_factory(
            state.features.shape[1],
            "Flat",
            {"inner": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}["l2"],
        )
        features_train = state.features[state.labeled_indices]
        print("standard, |features_train|", len(features_train))

        knn_index_train.train(features_train)
        knn_index_train.add(features_train)

        state.all_distances, state.all_indices = knn_index_train.search(
            state.features, k=30
        )

        state.label_array = state.label_array[state.labeled_indices]

    print("knn labeled", time.time() - time_start)
    distances, indices = (
        state.all_distances[state.labeled_indices],
        state.all_indices[state.labeled_indices],
    )
    # figure out neighbors of updated uids
    quicker_updates = int(
        os.getenv("QUICKER_UPDATES", 0)
    )  # 0 = off, 1 = display only, 2 = display + knn
    new_labeled_nn_idx: Set[int] | None = None
    new_labeled_nn_uids = None
    quicker_updates = (
        quicker_updates
        if (state.new_labeled_uids is not None and len(state.new_labeled_uids) > 0)
        else 0
    )
    if quicker_updates:
        # get indices of new_labeled_uids
        new_labeled_idx = data[data.uid.isin(state.new_labeled_uids)].index
        logger.info(f"new_labeled_uids={state.new_labeled_uids}")
        logger.info(f"new_labeled_idx={new_labeled_idx}")
        new_labeled_nn_idx = set(state.all_indices[new_labeled_idx].flatten())
        logger.info(f"new_labeled_nn_idx={len(new_labeled_nn_idx)}")
        new_labeled_nn_uids = data[data.index.isin(new_labeled_nn_idx)].uid
    #
    # optimize_weight_exponent_func(state, annotations, data, distances, indices)
    #
    if quicker_updates:
        if "score_possible" not in data:
            data["label_predicted"] = None
            data["scores_predicted"] = None
            data["label_possible"] = ""
            data["score_possible"] = None
        else:
            data.loc[list(new_labeled_nn_idx), "label_predicted"] = None
            data.loc[list(new_labeled_nn_idx), "scores_predicted"] = None
            data.loc[list(new_labeled_nn_idx), "label_possible"] = ""
            data.loc[list(new_labeled_nn_idx), "score_possible"] = None
    else:
        data["label_predicted"] = None
        data["scores_predicted"] = None
        data["label_possible"] = None
        data["score_possible"] = None
    # - make predictions for labeled indices
    logger.info(f"Using knn_rank_exponent={state.knn_rank_exponent}")
    indices_ = indices
    distances_ = distances
    labeled_indices_ = state.labeled_indices
    logger.info(f"labeled_indices_={len(labeled_indices_)}")
    if quicker_updates > 1:
        idx_sel = [
            i_
            for i_, idx_ in enumerate(state.labeled_indices)
            if idx_ in new_labeled_nn_idx
        ]
        indices_ = indices_[idx_sel]
        distances_ = distances_[idx_sel]
        labeled_indices_ = np.array(labeled_indices_)[idx_sel].tolist()
        logger.info(
            f"quicker_updates: labeled_indices_ before test_indices={len(labeled_indices_)}"
        )
        labeled_indices_.extend(test_indices)
        logger.info(f"quicker_updates: labeled_indices_={len(labeled_indices_)}")
    state.g_quick_status = "predicting labeled"
    make_predictions(
        annotations,
        data,
        indices_,
        distances_,
        state.label_array,
        labeled_indices_,
        None,
        test_indices,
        None,
        skip_first=True,
        knn_rank_exponent=state.knn_rank_exponent,
    )
    prev_near_labeled_perc = get_performance_key_val(
        state.performance_path, "percentage_near_labeled", -1.0
    )
    print("prev_near_labeled_perc", prev_near_labeled_perc)
    if prev_near_labeled_perc < min_prev_near_labeled_perc:
        state.g_quick_status = "computing most needed"
        counter_of_most_need, near_labeled_indices, near_labeled_perc = (
            compute_most_needed(
                state.all_indices, state.labeled_indices, state.features
            )
        )
        write_performance_key_val(
            state.performance_path, "percentage_near_labeled", near_labeled_perc
        )
    else:
        logger.warning(
            f"Skipping most needed because {prev_near_labeled_perc=}<{min_prev_near_labeled_perc}"
        )
        near_labeled_indices = np.arange(len(state.all_distances))
        counter_of_most_need = {}
    time_start = time.time()
    # distances, indices = knn_index.search(features[near_labeled_indices], k=k)
    distances, indices = (
        state.all_distances[near_labeled_indices],
        state.all_indices[near_labeled_indices],
    )
    logger.info(f"knn near_labeled={time.time() - time_start}")
    data["entropy"] = -np.inf
    data["al_measure"] = len(data) + 1
    data["most_needed"] = len(data) + 1
    # - make predictions for near labeled
    indices_ = indices
    distances_ = distances
    near_labeled_indices_ = near_labeled_indices
    if quicker_updates > 1:
        idx_sel = [
            i_
            for i_, idx_ in enumerate(near_labeled_indices)
            if idx_ in new_labeled_nn_idx
        ]
        indices_ = indices_[idx_sel]
        distances_ = distances_[idx_sel]
        near_labeled_indices_ = np.array(near_labeled_indices)[idx_sel].tolist()
        near_labeled_indices_.extend(test_indices)
    predicted_test = []
    true_test = []
    state.g_quick_status = "computing predictions"
    make_predictions(
        annotations,
        data,
        indices_,
        distances_,
        state.label_array,
        near_labeled_indices_,
        predicted_test,
        test_indices,
        true_test,
        knn_rank_exponent=state.knn_rank_exponent,
    )
    print("|predicted_test|", len(predicted_test))
    print("make_predictions end", time.time())
    data.label_predicted = data.label_predicted.apply(lambda x_: canon_(x_))
    data.label_true = data.label_true.apply(lambda x_: canon_(x_))
    # FRE
    state.g_quick_status = "computing FRE"
    compute_fre(state.label_array, data, state.features, state.labeled_indices)
    # history
    state.g_quick_status = "setting Most needed"
    #
    blurp = sorted(counter_of_most_need.items(), key=lambda t_: -t_[1])
    for i_, (most_needed_i, most_needed) in enumerate(blurp):
        # print("most_needed", most_needed, data.at[most_needed_i, "uid"])
        # assert (
        #     most_needed_i not in labeled_indices and most_needed_i not in test_indices
        # )
        data.at[most_needed_i, "most_needed"] = i_
        if i_ > 20:
            break

    #
    state.g_quick_status = "computing performance"
    compute_performance(predicted_test, true_test, state, annotations, data)
    state.g_quick_status = "coloring and labelling"
    color_and_label(
        data,
        annotations,
        display_update_uids=new_labeled_nn_uids
        if new_labeled_nn_uids is not None and len(new_labeled_nn_uids) > 0
        else None,
    )

    with open(state.doublecheck_path) as f:
        double_checked = set(json.load(f)["checked"])
        data["double_checked"] = data["uid"].apply(
            lambda x_: int(x_ in set(double_checked))
        )

    if "species_true" in data:
        data = data.drop("species_true", axis=1)
    if "species_predicted" in data:
        data = data.drop("species_predicted", axis=1)
    data.to_csv(os.path.join(state.data_folder, "annflux", "annflux.csv"), index=False)
    print("no prediction", len(data[(data.score_predicted == 0) & (data.labeled == 0)]))
    print("instant_reclassification done")
    state.g_quick_status = "idle"


def make_predictions(
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    indices,
    distances,
    train_labels,
    org_map: List[int],
    predicted_test,
    test_indices,
    true_test,
    skip_first=False,
    knn_rank_exponent=0.5,
):
    """

    :param annotations:
    :param data:
    :param indices: matrix with rows corresponding to predicted samples and columns to indices of neighbors in knn
    training set
    :param distances: matrix with rows corresponding to predicted samples and columns to distances to neighbors in knn
    training set
    :param train_labels: array with labels of knn training set
    :param org_map: maps index of (indices, distances) to original index
    :param predicted_test:
    :param test_indices: test_indices in original dataset
    :param true_test:
    :param skip_first:
    :param knn_rank_exponent:
    :return:
    """
    print(len(indices), len(distances), len(train_labels))
    # assert len(indices) == len(distances) == len(train_labels)
    custom_thresholds = {"Normal": 0.50, "Too dark": 0.5}
    tmp_counter = 0
    for i, indices_for_i in tqdm(enumerate(indices), desc="making knn predictions"):
        org_index = org_map[i]
        probabilities = defaultdict(lambda: 0)
        max_mass = 0
        for i2, multilabel_ in enumerate(train_labels[indices_for_i]):
            if skip_first and i2 == 0:
                continue
            if multilabel_ is not None:
                distance_weight = (
                    distances[i][i2] ** knn_rank_exponent
                )  # if knn_rank_exponent is None else ((2 + i2) ** knn_rank_exponent)
                for label_ in multilabel_:
                    probabilities[label_] += 1 / distance_weight
                if len(multilabel_) > 0:
                    max_mass += 1 / distance_weight
        for label_ in probabilities:
            probabilities[label_] /= max_mass
        tmp_counter += org_index in test_indices
        if len(probabilities) > 0:
            # max_label = list(probabilities.keys())[np.argmax(probabilities.values())]
            max_labels = [
                label_
                for label_, prob_ in probabilities.items()
                if prob_ > custom_thresholds.get(label_, 0.5)
            ]
            if "Normal" in max_labels and len(max_labels) > 1:
                max_labels.remove("Normal")
            if len(max_labels) == 0:
                max_index = np.argmax(list(probabilities.values()))
                max_labels = [list(probabilities.keys())[max_index]]
            possible_labels = [
                label_
                for label_, prob_ in sorted(
                    probabilities.items(), key=lambda t_: -t_[1]
                )
                if 0.01 < prob_ < 0.50
            ]
            # print(max_labels)
            data.at[org_index, "score_possible"] = ",".join(
                [f"{probabilities[label_]:.2f}" for label_ in possible_labels]
            )
            data.at[org_index, "label_possible"] = ",".join(possible_labels)
            if len(max_labels) > 0:
                data.at[org_index, "label_predicted"] = ",".join(max_labels)
                data.at[org_index, "score_predicted"] = min(
                    [probabilities[label_] for label_ in max_labels]
                )  # TODO
                data.at[org_index, "scores_predicted"] = ",".join(
                    [f"{probabilities[label_]:.2f}" for label_ in max_labels]
                )
                # entropy
                p = np.array(list(probabilities.values()))
                data.at[org_index, "entropy"] = -1 * (p * np.log2(p)).sum()
                #

                if org_index in test_indices and predicted_test is not None:
                    test_uid = data.at[org_index, "uid"]
                    if test_uid in annotations.keys():
                        # print("in test", max_label, annotations[test_uid])
                        predicted_test.append(max_labels)
                        true_test.append(annotations[test_uid].split(","))
                elif data.at[org_index, "uid"] in annotations:
                    data.at[org_index, "score_true"] = probabilities.get(
                        annotations[data.at[org_index, "uid"]], -1
                    )
            else:
                data.at[org_index, "label_predicted"] = None
                data.at[org_index, "score_predicted"] = 0
                data.at[org_index, "scores_predicted"] = None
    print("tmp_counter", tmp_counter)
