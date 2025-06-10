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
from typing import List, Dict

import numpy as np
import pandas
from sklearn.decomposition import PCA

from annflux.tools.data import canon_


def compute_fre(
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    features: np.array,
    labeled_indices: List[int],
    test_uids: List[str],
):
    """
    Compute feature reconstruction error
    """
    label_agg_array = np.array(
        [
            (annotations.get(uid) if annotations.get(uid) else None)
            if uid not in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ]
    )
    agg_to_indices = defaultdict(lambda: [])
    for index_ in labeled_indices:
        if label_agg_array[index_] is not None:
            agg_to_indices[canon_(label_agg_array[index_])].append(index_)
    agg_to_pca = {}
    time_start = time.time()
    for agg, indices_ in agg_to_indices.items():
        feat_ = features[sorted(indices_)]
        if len(feat_) > 10:
            pca = PCA(n_components=0.95)
            pca.fit(feat_)
            agg_to_pca[agg] = pca
    print("pca analysis took", time.time() - time_start)
    time_start = time.time()
    column_name = "fre"
    data[column_name] = None
    for label_predicted_ in data.label_predicted.unique():
        if pandas.isna(label_predicted_):
            continue
        indices_ = np.where(data.label_predicted == label_predicted_)[0]

        features_ = features[indices_]

        if label_predicted_ in agg_to_pca:
            pca: PCA = agg_to_pca[label_predicted_]
            fre = np.linalg.norm(
                pca.inverse_transform(pca.transform(features_)) - features_, axis=1
            )
            print(label_predicted_, fre.min(), fre.max())
            data.loc[indices_, column_name] = fre
    print("pca application took", time.time() - time_start)
    data[column_name] /= data[column_name].max()
    data[column_name] = 1 - data[column_name]
    data[column_name].fillna(0, inplace=True)
