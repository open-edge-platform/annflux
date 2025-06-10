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

import sys
sys.path.append(r"C:\Laurens\applications.ai.geti.annflux\src")

import json
import os
import shutil
from pathlib import Path

import flask
import pytest

from annflux.data.envdataset.data import SyntheticDataset
from annflux.scripts.annflux_cli import go_command
from annflux.ui.basic.run_server import _init, get_app

annflux_data_path: str | None = None


def create_app():
    #
    global annflux_data_path
    data_source = SyntheticDataset()
    data_source.download()
    data_folder = Path(os.path.expanduser("~/annflux/data/envdataset"))
    shutil.rmtree(data_folder)
    data_source.copy_to(data_folder)
    print("here")
    go_command(data_folder, ["No plant", "Flowering", "Vegetative"])
    annflux_folder = data_folder / "annflux"
    annflux_data_path = annflux_folder / "annflux.csv"
    os.environ["PROJECT_ROOT"] = str(data_folder)
    #
    app = get_app()
    app.config.from_mapping(TESTING=True)
    _init()
    return app


@pytest.fixture
def app():
    app: flask.Flask = create_app()

    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


def test_index(client):
    response = client.get("/")
    print(response.data)
    assert b"/static/annflux_layout.js" in response.data


def test_refresh(client):
    response = client.post("/label", json={})
    print(response.data)
    assert json.loads(response.data) == {"success": True}
