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
import numpy as np

def softmax(logits: np.ndarray, temperature: float = 1.0):
    """
    Compute softmax values for each sets of scores in x.
    :param logits: numpy array with dimensions (num_predictions, num_classes)
    :param temperature: softmax temperature
    :return: numpy array with same dimensions as input
    """
    in_dim = logits.ndim
    if in_dim == 1:
        logits = np.expand_dims(logits, axis=0)
    logits = logits.astype(np.longdouble)
    if temperature != 1.0:
        logits = logits.copy()
        logits *= 1.0 / temperature
    max_l = np.max(logits, axis=1)
    e_x = np.exp(logits - np.expand_dims(max_l, axis=1))
    result = e_x / np.expand_dims(e_x.sum(axis=1), axis=-1)
    return result if in_dim > 1 else result[0]
