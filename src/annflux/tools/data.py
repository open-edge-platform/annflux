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
import copy
import itertools
import json
import logging
import os
import time
from functools import lru_cache
from typing import List

import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

from annflux.tools.mixed import get_basic_logger


@lru_cache
def canon_(s_: str | list[str]) -> str | None:
    """
    Canonizes a multilabel string separated by comma's
    """
    return (
        ",".join(sorted(s_.split(","))) if isinstance(s_, str) else ",".join(sorted(s_))
        if s_ is not None and not pandas.isna(s_)
        else None
    )


def color_and_label(
    data,
    annotations,
    display_update_uids: List[str] = None,
    logger: logging.Logger = get_basic_logger("color_and_label"),
):
    start_time = time.time()
    logger.info(f"color_and_label start={time.time()}")
    #
    data_to_update = (
        data[data.uid.isin(display_update_uids)]
        if display_update_uids is not None
        else data
    )
    logger.info(f"color_and_label: updating {len(data_to_update)} media")
    #
    label_array = np.array(
        [annotations.get(uid) for i, uid in enumerate(data.uid.values)]
    )
    cmap = plt.get_cmap("viridis")
    cmap_distinct = plt.get_cmap("tab20b")
    annotation_values = set([canon_(x_) for x_ in annotations.values()])
    individual_labels = set(
        list(itertools.chain(*[x_.split(",") for x_ in annotations.values()]))
    )
    data_to_update["label_predicted"] = data_to_update["label_predicted"].apply(
        lambda x_: x_
        if len(
            set(
                x_.split(",") if x_ is not None and not pandas.isna(x_) else []
            ).intersection(individual_labels)
        )
        > 0
        else "n/a"
    )

    data["label_true"] = label_array
    labeled_uids = set(annotations.keys())
    data["labeled"] = data["uid"].apply(lambda x_: int(x_ in labeled_uids))

    data_to_update.label_predicted = data_to_update.label_predicted.apply(
        lambda x_: canon_(x_)
    )
    data.label_true = data.label_true.apply(lambda x_: canon_(x_))

    unique_labels = sorted(
        list(
            set(
                [
                    x_
                    for x_ in data["label_predicted"].unique()
                    if x_ is not None and x_ != "n/a"
                ]
            ).union(annotation_values)
        )
    )
    unique_labels.append("n/a")
    logger.info(f"unique_labels={unique_labels}")
    logger.info(f"unique_labels end={time.time() - start_time}")
    label_to_float = dict(
        list(zip(unique_labels, np.arange(len(unique_labels)) / len(unique_labels)))
    )

    time_start = time.time()
    prob_to_color = {k: rgb2hex(cmap(k / 100)) for k in range(101)}
    data_to_update["color_prob"] = data_to_update.apply(
        lambda x_: prob_to_color[int(x_.score_predicted * 100)]
        if not x_.labeled
        else "#800080",
        axis=1,
    )
    logger.info(f"color_prob took={time.time() - start_time}")
    if "fre" in data_to_update.columns:
        q5, q95 = np.percentile(data_to_update.fre, [1, 99])
        logger.info(f"fre={q5, q95}")
        if q5 != q95:
            data_to_update["fre_for_color"] = data_to_update.fre.apply(
                lambda x_: (np.clip(x_, q5, q95) - q5) / (q95 - q5)
            )
            data_to_update["color_fre"] = data_to_update.apply(
                lambda x_: prob_to_color[int(x_.fre_for_color * 100)]
                if not pandas.isna(x_.fre_for_color)
                else "",
                axis=1,
            )
            del data_to_update["fre_for_color"]
        logger.info(f"fre_for_color took={time.time() - start_time}")
    class_to_color = {
        k: rgb2hex(cmap_distinct(label_to_float[k]))
        for k in list(unique_labels)
        if k is not None
    }

    def class_to_color_(x_):
        return class_to_color[x_] if x_ not in ["n/a", None] else "#AAAAAA"

    data["color_class"] = data.apply(
        lambda x_: class_to_color_(
            x_.label_predicted if not x_.labeled else x_.label_true
        ),
        axis=1,
    )
    logger.info(f"color_class took={time.time() - start_time}")
    #
    if "label_possible" in data_to_update.columns:
        data_to_update["incorrect_score"] = 0.0
        for r, row in data_to_update.iterrows():
            if (
                row.labeled == 0
                or pandas.isna(row.label_possible)
                or pandas.isna(row.score_possible)
                or len(row.label_possible) == 0
            ):
                continue
            # print(row.label_possible, row.score_possible)
            predicted_map = dict(
                list(
                    zip(
                        row.label_possible.split(",")
                        if isinstance(row.label_possible, str)
                        else [],
                        map(float, row.score_possible.split(","))
                        if isinstance(row.score_possible, str)
                        else [row.score_possible],
                    )
                )
            )
            labels_true = set(row.label_true.split(","))
            for key in copy.copy(list(predicted_map.keys())):
                if predicted_map[key] < 0.5 and key not in labels_true:
                    del predicted_map[key]
            score = 0
            for label_ in set(predicted_map.keys()).union(labels_true):
                score += abs(
                    ((label_ in labels_true) * 1) - predicted_map.get(label_, 0.0)
                )
            data_to_update.at[r, "incorrect_score"] = score
        data_to_update.incorrect_score = (
            data_to_update.incorrect_score.max() - data_to_update.incorrect_score
        )
    #
    logger.info(f"label_possible took={time.time() - start_time}")
    logger.info(f"coloring took={time.time() - time_start}")


def remove_uids_from_double_check(
    uids: List[str],
    doublecheck_path,
    logger=get_basic_logger("remove_uids_from_double_check"),
):
    if not os.path.exists(doublecheck_path):
        return
    j_doublecheck = json.load(open(doublecheck_path))
    doublecheck_uids = set(j_doublecheck["checked"])
    num_before = len(doublecheck_uids)
    doublecheck_uids -= set(uids)
    num_after = len(doublecheck_uids)
    logger.info(f"remove_uids_from_double_check removed {num_before - num_after}")
    j_doublecheck["checked"] = list(doublecheck_uids)
    with open(doublecheck_path, "w") as f:
        json.dump(j_doublecheck, f, indent=2)


def get_images_path() -> str:
    """
    Get default path for images
    :return:
    """
    return os.path.join(get_project_root(), "images")


def get_project_root() -> str:
    """
    Get PROJECT_ROOT environment variable, raises ValueError if not set
    :return: PROJECT_ROOT
    """
    project_root = os.getenv("PROJECT_ROOT", None)
    if project_root is None:
        raise ValueError("PROJECT_ROOT not set")
    return project_root


def add_group_to_exclusivity(group_children: List[str], exclusivity_path: str):
    """
    Add an exclusivity group to the exclusivity database
    """
    exclusivity_relations = itertools.combinations(group_children, 2)
    update = pandas.DataFrame(exclusivity_relations, columns=["left", "right"])
    if os.path.exists(exclusivity_path):
        exclusivity_table = pandas.read_csv(exclusivity_path)
        exclusivity_table = pandas.concat(
            [
                exclusivity_table,
                update,
            ],
            ignore_index=True,
        )
        exclusivity_table.drop_duplicates(["left", "right"])
    else:
        exclusivity_table = update
    print(f"Adding {group_children} to {exclusivity_path}")
    exclusivity_table.to_csv(exclusivity_path, index=False)
