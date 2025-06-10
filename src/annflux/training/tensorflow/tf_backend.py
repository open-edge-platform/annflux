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
import logging
import math
import os
from collections import defaultdict, Counter
from typing import Dict, List

import numpy as np
from keras import Input, Model
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.src.layers import Dense, Lambda
from keras.src.legacy.backend import l2_normalize

from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from annflux.tools.core import AnnFluxState

logger = logging.getLogger("annflux_server")


def linear_retraining(state: AnnFluxState, status_callback):
    if state.labeled_indices is None or len(state.labeled_indices) == 0:
        return
    logger.info(f"{len(state.labeled_indices)=}")
    balance = True
    binarizer = MultiLabelBinarizer()
    no_label_for_labeled_idx = np.where(
        state.label_array[state.labeled_indices] == None  # noqa
    )[0]
    if len(no_label_for_labeled_idx) > 0:
        raise RuntimeError(
            f"no label for idx {np.array(state.labeled_indices)[no_label_for_labeled_idx]}"
        )
    binarizer.fit(state.label_array[state.labeled_indices])
    targets = binarizer.transform(state.label_array[state.labeled_indices])

    test_targets = binarizer.transform(
        state.label_array_test[state.labeled_test_indices]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        state.features[state.labeled_indices], targets, test_size=0.10, random_state=42
    )
    input_ = Input(shape=(state.features.shape[1],))
    dense = input_
    activation = "relu"
    features_ = Dense(state.features.shape[1], activation=activation, name="features")(
        dense
    )
    features_ = Lambda(lambda x: l2_normalize(x, axis=1))(features_)
    predictions = Dense(len(binarizer.classes_), activation="sigmoid")(features_)
    model = Model(inputs=[input_], outputs=[predictions])

    model2 = Model(inputs=[input_], outputs=[features_])
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=0.000001, verbose=1
    )
    model.summary()

    loss_ = "binary_crossentropy"
    model.compile(loss=loss_, optimizer="adam", metrics=["accuracy"])

    weights_path = os.path.join(state.working_folder, "linear.weights.h5")
    checkpointer = ModelCheckpoint(
        weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=True,
    )

    model.fit(
        x=BalanceSequence(x_train, y_train, 1024, balance),
        batch_size=1024,
        validation_data=(x_test, y_test),
        epochs=200,
        verbose=1,
        callbacks=[
            reduce_lr,
            checkpointer,
            EarlyStopping(patience=5),
            status_callback,
        ],
    )
    model.load_weights(weights_path)
    test_predictions = model.predict(state.features[state.labeled_test_indices])
    acc_test = accuracy_score(test_targets, (test_predictions > 0.5).astype(int))
    logger.info(f"linear from features acc = {acc_test}")

    state.g_quick_status = "recomputing features"
    state.features = model2.predict(state.features)

    return weights_path


class BalanceSequence(PyDataset):
    def __init__(self, x_set, y_set, batch_size, balance: bool = False):  # noqa
        self.x, self.y = np.array(x_set), np.array(y_set)
        class_counts = Counter(np.argmax(self.y, axis=1))
        self.classes_ = list(class_counts.keys())
        if balance:
            self.class_weights = None
        else:
            self.class_weights = np.array(
                [class_counts[x_] for x_ in self.classes_], dtype=float
            )
            self.class_weights /= self.class_weights.sum()
        self.class_to_indices: Dict[int, List[int]] = defaultdict(lambda: [])
        for i_, class_ in enumerate(np.argmax(self.y, axis=1)):
            self.class_to_indices[class_].append(i_)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices_batch = []
        for _ in range(self.batch_size):
            class_ = np.random.choice(self.classes_, p=self.class_weights)
            indices_batch.append(np.random.choice(self.class_to_indices[class_]))

        print(type(self.x[indices_batch]))
        return self.x[indices_batch], self.y[indices_batch]
