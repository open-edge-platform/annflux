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
import hashlib
import os
from typing import List
import numpy as np
import pandas
from tqdm import tqdm


def numpy_load(
    path,
    key,
    check_for_split_format: bool = True,
    split_indices=None,
    select_per_split: List[List[int]] = None,
) -> np.ndarray:
    """
    Load implementation of numpy.load that supports split files
    """

    if not os.path.exists(path) and check_for_split_format:
        path = split_template(path)
    is_split_template = "{split_index}" in path

    if not is_split_template:
        array = np.load(path)[key]
    else:
        if split_indices is None:
            paths_to_load = get_part_paths(path)
        else:
            paths_to_load = [path.format(split_index=i_) for i_ in split_indices]

        array = []
        for split_index, part_path in tqdm(
            enumerate(paths_to_load), desc=f"Reading {path}"
        ):
            data: np.array = np.load(part_path)[key]
            if select_per_split is not None:
                # noinspection PyTypeChecker
                data = data[select_per_split[split_index]]
                print("|data", len(data))
            array.append(data)

        array = np.vstack(array)
    return array


def split_template(path) -> str:
    """
    Returns the split file template for a path consisting of basename.{split_index}.ext
    """
    basename, extension = os.path.splitext(path)
    return basename + ".{split_index}" + extension


def get_part_paths(path):
    split_index = 0
    paths_to_load = []
    while os.path.exists(path.format(split_index=split_index)):
        paths_to_load.append(path.format(split_index=split_index))
        split_index += 1
    return paths_to_load


def read_table_pandas(
    filename, check_for_split_format=True, split_indices=None
) -> pandas.DataFrame:
    if not os.path.exists(filename) and check_for_split_format:
        filename = split_template(filename)
    is_split_template = "{split_index}" in filename

    if not is_split_template:
        table = pandas.read_csv(filename)
    else:
        if split_indices is None:
            paths_to_load = get_part_paths(filename)
        else:
            paths_to_load = [filename.format(split_index=i_) for i_ in split_indices]

        tables = []
        for part_path in tqdm(paths_to_load, desc=f"loading {filename}"):
            tables.append(pandas.read_csv(part_path))
        table = pandas.concat(tables, axis=0)
    return table


def create_directory(*path):
    if len(path) > 1:
        path = os.path.join(path[0], *path[1:])
    else:
        path = path[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


def file_hash(path):
    """
    Computes a hash from a file.
    :param path:
    :return:
    """
    block_size = 65536
    hasher = hashlib.sha224()
    with open(path, "rb") as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()
