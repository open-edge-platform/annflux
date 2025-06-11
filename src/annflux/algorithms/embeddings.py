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
import os

import numpy as np
import umap
from sklearn.manifold import TSNE

tsne_method = os.getenv("TSNE_METHOD", "umap")


def normalize_and_scale(embedding):
    embedding -= np.min(embedding, axis=0, keepdims=True)
    embedding /= np.max(embedding, axis=0, keepdims=True)
    embedding *= 40
    embedding -= 20
    return embedding


def compute_tsne_scikit(features):
    embedding = TSNE(
        n_components=2, init="random", perplexity=3, verbose=True
    ).fit_transform(features.astype(float))

    return normalize_and_scale(embedding)


def compute_umap(features):
    embedding = umap.UMAP().fit_transform(features)
    return normalize_and_scale(embedding)


if tsne_method == "umap":
    compute_tsne = compute_umap
else:
    compute_tsne = compute_tsne_scikit
