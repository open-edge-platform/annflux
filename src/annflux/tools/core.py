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
import time
from pathlib import Path
from typing import Optional

import numpy as np
from numpy._typing import NDArray


class AnnFluxState(object):
    """
    Stores the core state of the AnnFlux state
    """
    def __init__(self, working_folder):
        super().__init__()
        self.time_new_status_time = None
        self.working_folder_ = Path(working_folder)

    @property
    def data_folder(self) -> str:
        return str(self.working_folder_ / "..")

    @property
    def working_folder(self) -> str:
        return str(self.working_folder_)

    @property
    def timings_path(self) -> str:
        return str(self.working_folder_ / "timings.csv")

    @property
    def g_quick_status(self):
        return self.g_quick_status_

    @g_quick_status.setter
    def g_quick_status(self, val):
        if val != self.g_quick_status_:
            self.time_new_status_time = time.time()
        self.g_quick_status_ = val
        timings_path = self.working_folder_ / "timings.csv"
        if not timings_path.exists():
            self.write_timing(
                timings_path, "status", "timestamp", "num_total", "num_labeled", "w"
            )
        self.write_timing(
            timings_path,
            self.g_quick_status_,
            time.time(),
            len(self.features) if self.features is not None else None,
            len(self.labeled_indices) if self.labeled_indices is not None else None,
        )

    @staticmethod
    def write_timing(timings_path, key, val, num_total, num_labeled, mode="a"):
        with open(timings_path, mode) as f:
            f.write(f"{key},{val},{num_total or ''},{num_labeled or ''}\n")

    cache_for: str = None
    features: np.array = None
    knn_index = None
    all_distances = None
    all_indices = None
    opt_knn_rank_exponent = None
    knn_rank_exponent = 3
    label_array: NDArray[list[str]] | None = None
    label_array_test = None
    labeled_indices = None
    labeled_test_indices = None
    version_for_recompute = None
    g_quick_status_ = None
    new_labeled_uids = None
    train_thread = None
    trained_for_version_pre = None
    linear_status_epoch: int = None
    trained_for_version: int = None
    optimize_weight_exponent: bool = False
    labels_path: str = None
    doublecheck_path: str = None
    performance_path: str = None
