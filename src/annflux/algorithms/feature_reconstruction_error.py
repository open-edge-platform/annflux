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
from collections import defaultdict
from typing import List

import numpy as np
import pandas
from sklearn.decomposition import PCA

from annflux.tools.data import canon_
from annflux.tools.mixed import get_logger_by_name


def compute_fre(
    label_agg_array,
    data: pandas.DataFrame,
    features: np.array,
    labeled_indices: List[int],
):
    """
    Compute feature reconstruction error
    """
    logger = get_logger_by_name("algorithms")
    time_start = time.time()
    agg_to_indices = defaultdict(lambda: [])
    for index_ in labeled_indices:
        annotation_ = label_agg_array[index_]
        if annotation_ is not None:
            agg_to_indices[canon_(",".join(annotation_))].append(index_)  # noqa
    agg_to_pca = {}
    logger.info(f"[TIMING] PCA data preparation took {time.time() - time_start} s")
    time_start = time.time()
    for agg, indices_ in agg_to_indices.items():
        feat_ = features[sorted(indices_)]
        if len(feat_) > 10:
            pca = PCA(n_components=0.95)
            pca.fit(feat_)
            agg_to_pca[agg] = pca
    logger.info(f"[TIMING] PCA model computation took {time.time() - time_start} s")
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
    logger.info(f"[TIMING] FRE computation took {time.time() - time_start} s")
    time_start = time.time()
    data[column_name] /= data[column_name].max()
    data[column_name] = 1 - data[column_name]
    data[column_name].fillna(0, inplace=True)
    logger.info(f"[TIMING] FRE setting in DataFrame took {time.time() - time_start} s")
