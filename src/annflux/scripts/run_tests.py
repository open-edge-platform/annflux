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
import os.path
import shlex
import shutil
from tempfile import mkdtemp

from annflux.data.envdataset.data import EnvDataset
# see: https://pythontest.com/testing-argparse-apps/ for inspiration
from annflux.scripts.annflux_cli import execute

test_type = os.getenv("TEST_TYPE", "user")

valid_test_types = [
    "user",  # run by end-user on CLI
    "pytest",  # run as part of test runner
    "nightly-short",  # run in nightly CI
]
if test_type not in valid_test_types:
    raise ValueError(f"Unknown TEST_TYPE should be one of {valid_test_types}")


def _test_go(folder, start_labels):
    print("Testing go command")
    command_ = f"go {folder} --start_labels {start_labels}"
    print(f"{command_=}")
    if test_type != "pytest":
        os.system(f"annflux {command_}")
    else:
        execute(shlex.split(command_))


def _test_train_then_features(test_data_path):
    print("Testing train_then_features command")
    command_ = f"train_then_features {test_data_path} --num_epochs {1 if test_type != 'nightly-short' else 10}"
    if test_type != "pytest":
        os.system(f"annflux {command_}")
    else:
        execute(shlex.split(command_))


def _test_export(test_data_path) -> str:
    print("Testing export command")
    model_package_folder = os.path.join(mkdtemp(), "package")
    # TODO: remove
    command_ = f"export {test_data_path} {model_package_folder}"
    if test_type != "pytest":
        os.system(f"annflux {command_}")
    else:
        execute(shlex.split(command_))
    return str(model_package_folder)


def run_cli_tests():
    print("START")
    data_source = EnvDataset()
    data_source.download()
    test_data_path = os.path.expanduser("~/annflux/data/envdataset")
    shutil.rmtree(test_data_path)
    data_source.copy_to(test_data_path)
    annflux_dir = os.path.join(test_data_path, "annflux")
    if os.path.isdir(annflux_dir):
        shutil.rmtree(annflux_dir)
    try:
        _test_go(test_data_path, ["A", "B", "C"])
        assert os.path.isdir(
            os.path.expanduser("~/annflux/data/envdataset/annflux")
        )
    except:  # noqa
        print("test_go failed")
        raise
    #
    try:
        shutil.copy(
            data_source.true_labels_path, os.path.join(annflux_dir, "labels.json")
        )
        _test_train_then_features(test_data_path)
    except:  # noqa
        print("test_train_then_features failed")
        raise
    # UI
    if test_type == "user":
        try:
            _test_ui(test_data_path)
        except:  # noqa
            print("test_ui failed")
            raise
    else:
        print("Skipping _test_ui because not in interactive mode")
    # export
    try:
        _test_export(test_data_path)
    except:  # noqa
        print("test_export failed")
        raise
    print("END")


def _test_ui(test_data_path):
    print("Testing basic_ui command")
    os.system(f"basic_ui {test_data_path}")


if __name__ == "__main__":
    run_cli_tests()
