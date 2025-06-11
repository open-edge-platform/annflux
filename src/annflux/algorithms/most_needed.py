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
import time
from collections import defaultdict
from typing import List

import numpy as np
import pandas


def     compute_most_needed(
    all_nn_indices: np.array,
    data: pandas.DataFrame,
    k: int,
    label_array: np.array,
    labeled_indices: List[int],
    test_indices: List[int],
    iterations=9,
    verbose: int = 0,
):
    """ """
    set_labeled_indices = set(labeled_indices)
    not_near_labeled_indices = []
    near_labeled_indices = []
    counter_of_most_need = defaultdict(lambda: 0)
    counter_of_most_need2 = defaultdict(lambda: 0)
    first_labeled_nn = []
    nth_labeled_nn = []
    seen_ = set()
    for index in set(list(range(len(data)))) - set_labeled_indices:
        nn_indices = all_nn_indices[index]
        # if none of the neighbors nn are labeled add the nn to most needed
        if set_labeled_indices.intersection(set(nn_indices)) == set():
            not_near_labeled_indices.append(index)
            for index_ in nn_indices:
                counter_of_most_need[index_] += 1
        else:
            seen_.add(index)
            near_labeled_indices.append(index)
            # find out which of the nn have labels
            labeled_nn_indices = np.where(label_array[nn_indices] != None)[0]  # noqa
            first_labeled_nn.append(labeled_nn_indices[0])
            nth_labeled_nn.append(
                labeled_nn_indices[14] if len(labeled_nn_indices) >= 15 else k
            )
            for i_, index_ in enumerate(nn_indices):
                if i_ > 15:
                    if index_ not in set_labeled_indices and index_ not in test_indices:
                        counter_of_most_need2[index_] += k - i_
    if verbose:
        print("np.mean(first_labeled_nn)", np.mean(first_labeled_nn))
        print("np.mean(nth_labeled_nn)", np.mean(nth_labeled_nn))
        print("fraction with 15 NN", (np.array(nth_labeled_nn) < k).mean())
        if len(counter_of_most_need2) > 0:
            print(
                "counter_of_most_need2",
                max(counter_of_most_need2.values()),
                min(counter_of_most_need2.values()),
            )
    # simulate adding the most needed image one by one to create diversity in most needed
    start_time = time.time()
    most_needed_ids = []
    tmp_set_labeled_indices = set_labeled_indices
    for it in range(iterations):
        print(f"Most needed iteration {it}")
        counter_of_most_need = defaultdict(lambda: 0)
        for index in (
            set(list(range(len(data)))) - tmp_set_labeled_indices
        ):
            if index in seen_:
                continue
            nn_indices = all_nn_indices[index]
            if tmp_set_labeled_indices.intersection(set(nn_indices)) == set():
                for index_ in nn_indices:
                    counter_of_most_need[index_] += 1
            else:
                seen_.add(index)
        if len(counter_of_most_need) == 0:
            break
        most_needed_id = sorted(counter_of_most_need.items(), key=lambda t_: -t_[1])[0][
            0
        ]
        tmp_set_labeled_indices.add(most_needed_id)
        most_needed_ids.append(most_needed_id)
    counter_of_most_need = defaultdict(lambda: 0)
    for i_ in most_needed_ids:
        counter_of_most_need[i_] = 1
    #
    if verbose:
        print("most_needed_ids", most_needed_ids)

        if len(counter_of_most_need) > 0:
            print(
                "counter_of_most_need",
                max(counter_of_most_need.values()),
                min(counter_of_most_need.values()),
            )
        print("most_needed_ids took", time.time() - start_time)

    perc_near_labeled = (len(near_labeled_indices) + len(set_labeled_indices)) / len(
        data
    )
    return counter_of_most_need, near_labeled_indices, perc_near_labeled
