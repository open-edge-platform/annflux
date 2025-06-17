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
import os
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Optional, List

import annflux


def remove_sys(list_: Optional[List[str]]):
    """
    Removes labels starting with sys from list_
    """
    if list_ is None:
        return list_
    filtered_list = []
    for x_ in list_:
        if "sys:" not in x_:
            filtered_list.append(x_)
    return filtered_list


def get_logger_by_name(name: str, mode: str = "a", level=logging.INFO):
    return get_logger(os.path.join(os.getenv("LOG_FOLDER", "."), f"{name}.log"), mode, level)


def get_logger(filename: str, mode: str = "a", level=logging.INFO) -> logging.Logger:
    """
    Enable file and console logging to 'filename'
    :param filename:
    :param mode: file mode ('a','w', etc.) for log file
    :param level:
    :return:
    """
    logger = logging.getLogger("")
    logger.setLevel(level)
    logger.handlers = []
    file_logger, formatter = create_file_logger(filename, mode)
    logger.addHandler(file_logger)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)

    return logger


def create_file_logger(filename, mode):
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_logger = logging.FileHandler(filename, mode=mode)
    file_logger.setFormatter(formatter)
    file_logger.setLevel(logging.DEBUG)
    return file_logger, formatter


def get_basic_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    return logger


_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}


def str2bool(value, raise_exc=False):
    result = None
    value_ = str(value).lower()
    if value_ in _true_set:
        result = True
    elif value_ in _false_set:
        result = False
    else:
        if raise_exc:
            raise ValueError('Expected "{}"'.format('", "'.join(_true_set | _false_set)))
    return result


def get_version():
    """
    Get version of annflux
    """
    below_root = Path(os.path.dirname(__file__)) / ".." / ".." / ".."
    is_package = "pyproject.toml" not in os.listdir(below_root)
    return {
        "version": version("annflux")
        if is_package
        else tomllib.load(open(below_root / "pyproject.toml", "rb"))["project"][
            "version"
        ]
    }
