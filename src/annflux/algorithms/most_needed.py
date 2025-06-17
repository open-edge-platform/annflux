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
import logging
import time
from collections import defaultdict
from typing import List

import numpy as np
from faiss import pairwise_distances
from tqdm import tqdm

logger = logging.getLogger("annflux_server")


def diversify(
    features_sorted: np.array, diversify_from: int = None, fraction_others=0.1
) -> list[int]:
    if diversify_from is None:
        diversify_from = len(features_sorted)
    result: list[int] = [0]

    for _ in tqdm(range(diversify_from), total=diversify_from, desc="diversifying"):
        others = list(set(range(len(features_sorted))) - set(result))
        others = np.random.choice(
            others, size=int(fraction_others * len(others)), replace=False
        )  # TODO(improvement): use most needed as weights
        others = list(sorted(others))
        distances = pairwise_distances(
            features_sorted[list(sorted(result))], features_sorted[others]
        )
        sum_distances = np.sum(distances, axis=0)
        result.append(np.argmin(sum_distances))

    return result


def compute_most_needed(
    all_nn_indices: np.array, labeled_indices: List[int], all_features: np.array
):
    set_labeled_indices = set(labeled_indices)
    not_near_labeled_indices = []
    near_labeled_indices = []
    counter_of_most_need: dict[int, int] = defaultdict(
        lambda: 0
    )  # all_index -> most needed
    seen_ = set()
    num_total = len(all_nn_indices)
    indices_todo = set(list(range(num_total))) - set_labeled_indices
    start_time = time.time()
    for index in tqdm(
        indices_todo, total=len(indices_todo), desc="computing most needed"
    ):
        nn_indices = all_nn_indices[index]
        # if none of the neighbors nn are labeled add the nn to most needed
        if set_labeled_indices.intersection(set(nn_indices)) == set():
            not_near_labeled_indices.append(index)
            for index_ in nn_indices:
                counter_of_most_need[index_] += 1
        else:
            seen_.add(index)
            near_labeled_indices.append(index)
    #  improve diversity
    if len(counter_of_most_need) > 0:
        most_needed_i: list[int] = list(sorted(counter_of_most_need.keys()))
        diversified = diversify(
            all_features[most_needed_i],
            diversify_from=50,  # TODO(improvement): configurable
        )
        logger.info(f"{diversified=}")
        all_needed_i = np.array(most_needed_i)[diversified].tolist()
        counter_of_most_need = {}
        for i2, i in enumerate(all_needed_i):
            counter_of_most_need[i] = len(all_needed_i) - i2
    #

    logger.info(f"{len(counter_of_most_need)=}")
    if len(counter_of_most_need) > 0:
        logger.info(
            f"counter_of_most_need={max(counter_of_most_need.values()), min(counter_of_most_need.values())}"
        )
    logger.info(f"most_needed_ids took={time.time() - start_time}")

    perc_near_labeled = (
        len(near_labeled_indices) + len(set_labeled_indices)
    ) / num_total
    return counter_of_most_need, near_labeled_indices, perc_near_labeled
