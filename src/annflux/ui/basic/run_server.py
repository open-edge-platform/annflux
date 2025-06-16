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
import json
import logging
import os
import sys
import threading
import time
import tomllib
from datetime import datetime
from functools import update_wrapper, wraps
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flask
import numpy as np
import pandas
from flask import make_response, render_template, request, send_file
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.callbacks import Callback

import annflux
from annflux.algorithms.embeddings import compute_tsne
from annflux.tools.core import AnnFluxState
from annflux.tools.data import (
    add_group_to_exclusivity,
    get_images_path,
    remove_uids_from_double_check,
)
from annflux.tools.mixed import get_logger, str2bool
from annflux.training.annflux.quick import quick_reclassification
from annflux.training.tensorflow.tf_backend import linear_retraining

project_root: Optional[str] = None
images_path: Optional[str] = None
working_folder: Optional[str] = None
exclusivity_path: Optional[str] = None
label_defs_path: Optional[str] = None
label_provider_path: Optional[str] = None
g_state: Optional[AnnFluxState] = None
logger: Optional[logging.Logger] = None


class NoStatus(logging.Filter):
    def filter(self, record):
        return "POST /status" not in record.getMessage()


def _init():
    global \
        project_root, \
        images_path, \
        working_folder, \
        g_state, \
        exclusivity_path, \
        label_provider_path, \
        label_defs_path
    global logger
    project_root = os.getenv("PROJECT_ROOT", None)
    if project_root is None:
        raise RuntimeError("You should set PROJECT_ROOT environment variable")
    else:
        images_path = get_images_path()
        working_folder = os.path.join(project_root, "annflux")

    g_state = AnnFluxState(working_folder)

    g_state.doublecheck_path = os.path.join(
        g_state.data_folder, "annflux", "doublecheck.json"
    )
    exclusivity_path = os.path.join(g_state.data_folder, "annflux", "exclusivity.csv")
    label_provider_path = os.path.join(
        g_state.data_folder, "annflux", "label_provider.csv"
    )
    g_state.labels_path = os.path.join(g_state.data_folder, "annflux", "labels.json")
    label_defs_path = os.path.join(g_state.data_folder, "annflux", "label_defs.json")
    g_state.performance_path = os.path.join(
        g_state.data_folder, "annflux", "performance.json"
    )

    print(os.getenv("LOGGING_LEVEL", "INFO"))
    log_level: int = logging.getLevelName(os.getenv("LOGGING_LEVEL", "INFO"))
    log_path = os.path.join(g_state.working_folder, "annflux.log")
    os.makedirs(g_state.working_folder, exist_ok=True)
    logger = get_logger(log_path, level=log_level)
    logging.getLogger("werkzeug").addFilter(NoStatus())
    logger.warning(
        f"Logging to {log_path} with level {logging.getLevelName(log_level)}"
    )
    if os.getenv("LOG_FOLDER") is None:
        os.environ["LOG_FOLDER"] = g_state.working_folder
    print(f"{logger=}")
    print(f"Using project_root={project_root}, images_path ={images_path}")


app = flask.Flask(
    __name__,
    static_url_path=os.getenv("STATIC_URL", "/static"),
)


def get_app():
    return app


knn_type = "quick"
dump_linear_features = False
optimize_weight_exponent = False

num_unlabeled_certain = None


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Last-Modified"] = datetime.now()
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return update_wrapper(no_cache, view)


@app.route("/annflux")
@app.route("/")
def annflux_endpoint():
    """ """
    return render_template(
        {"annflux": "html", "golden": "annflux_golden.html"}[
            os.getenv("INDEED_TEMPLATE", "golden")
        ]
    )


@app.route("/data")
@nocache
def data_get():
    """
    Identification from image(s) (with authentication)
    :return: json server response
    """
    annflux_data_path = os.path.join(g_state.data_folder, "annflux", "annflux.csv")
    logger.info(f"annflux_data_path = {annflux_data_path}")
    return send_file(
        annflux_data_path,
        mimetype="text_csv",
        as_attachment=False,
    )


@app.route("/images/thumbnail/<uid>")
def thumbnail(uid):
    """ """
    thumb_path = os.path.join(images_path, f"{uid}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/jpg", as_attachment=False)


class StatusUpdate(Callback):
    def __init__(self, state: AnnFluxState):
        super().__init__()
        self.state = state

    def on_epoch_end(self, epoch, logs=None):
        self.state.linear_status_epoch = epoch


@app.route("/label_defs", methods=["PUT"])
def label_defs_add():
    """
    Stores label in label definitions file if it does not already exist

    Expects as input a tuple of the form [new_label, parent_label] or [new_label, parent_label, exclusive_under_parent]

    If exclusive_under_parent is True all the children of parent_label will be made mutually exclusive
    @return:
    """
    label_def = request.get_json(force=True)
    print("label_def", label_def)
    label_definitions: Dict[str, List[Tuple[str, str]]]  # Tuple = (label, parent)
    if not os.path.exists(label_defs_path):
        label_definitions = {"labels": []}
    else:
        label_definitions = json.load(open(label_defs_path))
    if label_def not in label_definitions["labels"]:
        label_definitions["labels"].append(label_def)
        # -- undetermined labels
        if os.path.exists(g_state.labels_path):
            modified_uids = []
            annotations = json.load(open(g_state.labels_path))
            exclusive_under_parent = False
            if len(label_def) == 2:
                new_label, parent_label = label_def
            elif len(label_def) == 3:
                new_label, parent_label, exclusive_under_parent = label_def
            else:
                raise RuntimeError()
            has_parent = parent_label not in ["null", "", None]
            # - assign undetermined state appropriate images
            for image_id, annotation in annotations.items():
                labels = annotation.split(",")
                if has_parent:
                    if parent_label in labels:
                        annotation += f",{new_label}=?"
                        annotations[image_id] = annotation
                        modified_uids.append(image_id)
                else:
                    annotation += f",{new_label}=?"
                    annotations[image_id] = annotation
                    modified_uids.append(image_id)
            json.dump(annotations, open(g_state.labels_path, "w"), indent=2)
            # remove uids from doublecheck list so that they appear for the viewer to check
            remove_uids_from_double_check(modified_uids, g_state.doublecheck_path)
            # exclusive_under_parent
            if exclusive_under_parent and has_parent:
                group_children = [
                    t_[0] for t_ in label_definitions["labels"] if t_[1] == parent_label
                ]
                add_group_to_exclusivity(group_children, exclusivity_path)

        # add new label to label definitions
        with open(label_defs_path, "w") as f:
            json.dump(label_definitions, f)
    return {"result": "ok"}


@app.route("/version")
def version_endpoint():
    return g_version


def get_version():
    """
    Get version of annflux
    """
    below_root = Path(os.path.dirname(annflux.__file__)) / ".." / ".."
    is_package = "pyproject.toml" not in os.listdir(below_root)
    return {
        "version": version("annflux")
        if is_package
        else tomllib.load(open(below_root / "pyproject.toml", "rb"))["project"][
            "version"
        ]
    }


g_version = get_version()


@app.route("/label", methods=["POST"])
def label():
    g_state.new_labeled_uids = set()
    if os.path.exists(g_state.labels_path):
        with open(g_state.labels_path, "r") as f:
            j_labels = json.load(f)
    else:
        j_labels = {}
    label_update = request.get_json(force=True)
    logger.info(f"label_update={label_update}")
    # double check
    if os.path.exists(g_state.doublecheck_path):
        with open(g_state.doublecheck_path, "r") as f:
            j_doublecheck = json.load(f)
    else:
        j_doublecheck = {"checked": []}
    for uid in label_update:
        if uid in j_labels:
            print(f"Adding {uid} to double check")
            j_doublecheck["checked"].append(uid)
        g_state.new_labeled_uids.add(uid)
    with open(g_state.doublecheck_path, "w") as f:
        json.dump(j_doublecheck, f, indent=2)  # noqa
    #
    # TODO: check that not incidentally undetermined labels are removed
    j_labels.update({k: v for k, v in label_update.items() if v != "n/a" and v != ""})
    with open(g_state.labels_path, "w") as f:
        json.dump(j_labels, f, indent=2)
    print(f"/label: {logger=}")
    quick_reclassification(g_state, logger=logger)
    return {
        "success": True,
    }


@app.route("/performance")
def performance():
    return (
        json.load(open(g_state.performance_path))
        if os.path.exists(g_state.performance_path)
        else {}
    )


@app.route("/detailed_performance/data")
def detailed_performance_data():
    return send_file(
        os.path.join(g_state.working_folder, "detailed_performance.csv"),
        mimetype="text_csv",
        as_attachment=False,
    )


@app.route("/exclusivity/data")
def exclusivity_data():
    if not os.path.exists(exclusivity_path):
        pandas.DataFrame(data={}, columns=["left", "right"]).to_csv(
            exclusivity_path, index=False
        )

    return send_file(exclusivity_path, mimetype="text_csv", as_attachment=False)


@app.route("/label_provider/data")
def label_provider_data():
    return send_file(label_provider_path, mimetype="text_csv", as_attachment=False)


@app.route("/exclusivity/data", methods=["POST"])
def exclusivity_data_post():
    pandas.DataFrame(data=request.get_json(), columns=["left", "right"]).to_csv(
        exclusivity_path, index=False
    )
    return {"success": True}


@app.route("/exclusivity")
def exclusivity_ui():
    return render_template("exclusivity.html")


@app.route("/label_provider")
def label_provider_ui():
    return render_template("label_provider.html")


@app.route("/label_defs")
def label_defs_list():
    return json.load(open(label_defs_path))


@app.route("/detailed_performance")
def detailed_performance():
    return render_template("detailed_performance.html")


def retrain_job(state: AnnFluxState):
    state.g_quick_status = "training"
    linear_retraining(state, StatusUpdate(state))
    #
    state.g_quick_status = "computing TSNE"
    embedding = compute_tsne(state.features)
    state.g_quick_status = "computing TSNE done"
    data = pandas.read_csv(
        os.path.join(state.data_folder, "annflux", "annflux.csv"),
        dtype={"label_predicted": str, "score_true": float},
    )
    data["e_0"] = embedding[:, 0]
    data["e_1"] = embedding[:, 1]
    data.to_csv(os.path.join(state.data_folder, "annflux", "annflux.csv"), index=False)
    state.trained_for_version_pre = len(state.labeled_indices)
    quick_reclassification(state, logger=logger)
    print("done", state.trained_for_version)
    state.trained_for_version = len(state.labeled_indices)


time_estimators = []
states = []


def estimate_duration(annflux_state: AnnFluxState, step_state: Tuple[str, int, int]):
    global time_estimators, states
    if len(time_estimators) == 0:
        timings = pandas.read_csv(annflux_state.timings_path)
        timings.dropna(inplace=True)
        timings.reset_index(drop=True, inplace=True)
        states = []
        transitions = []
        durations = []
        for r, row in timings.iterrows():
            if (
                r > 0
                and row.status != "idle"
                and timings.loc[r - 1, "status"] != "idle"
            ):
                # state = f"{timings.loc[r - 1, 'status']}-{row.status}"
                state = timings.loc[r - 1, "status"]
                if state not in states:
                    states.append(state)
                transitions.append(
                    (
                        states.index(state),
                        float(row.num_total),
                        float(row.num_labeled),
                    )
                )
                durations.append(
                    row.timestamp - timings.loc[r - 1, "timestamp"],
                )
        logger.debug(f"estimate_duration: transitions={transitions[:-10]}")
        x = np.vstack(transitions)

        # time_estimators = []
        for _ in range(10):
            rf = RandomForestRegressor()
            rf.fit(x, np.array(durations))
            time_estimators.append(rf)

    if step_state[0] in states:
        in_features = (
            np.array(
                [
                    states.index(step_state[0]),
                    float(step_state[1]),
                    float(step_state[2]),
                ]
            )
            .reshape(-1, 1)
            .T
        )
        results = []
        for rf_ in time_estimators:
            results.append(rf_.predict(in_features)[0])
        result = np.mean(results), np.std(results)
    else:
        result = 0, 0
    return result


@app.route("/status", methods=["POST"])
def status():
    label_update = request.get_json(force=True)
    auto_linear_train_idle_time = int(os.getenv("AUTO_LINEAR_TRAIN_IDLE_TIME", 1800))
    if label_update["idleTime"] > auto_linear_train_idle_time:
        if g_state.train_thread is None or not g_state.train_thread.is_alive():
            if g_state.labeled_indices is not None:
                logger.info(
                    f"Training from status: {g_state.trained_for_version=}"
                    f", {len(g_state.labeled_indices)=}, {label_update['idleTime']}"
                )
                if g_state.trained_for_version != len(g_state.labeled_indices):
                    g_state.train_thread = threading.Thread(
                        target=retrain_job, args=(g_state,)
                    )
                    g_state.train_thread.start()
                    pass
    #
    estimated_duration_s = 0
    duration_std_s = 0
    if g_state.features is not None and g_state.labeled_indices is not None:
        try:
            estimated_duration_s, duration_std_s = estimate_duration(
                g_state,
                (
                    g_state.g_quick_status,
                    len(g_state.features),
                    len(g_state.labeled_indices),
                ),
            )
        except ValueError:
            estimated_duration_s = 0
            duration_std_s = 0
    #
    status_duration = (
        time.time() - g_state.time_new_status_time
        if g_state.time_new_status_time is not None
        else 0
    )
    detailed_performance_path = os.path.join(
        g_state.working_folder, "detailed_performance.csv"
    )
    if os.path.exists(detailed_performance_path):
        detailed_performance_ = pandas.read_csv(detailed_performance_path)
        num_unlabeled_certain = int(detailed_performance_.num_predicted_certain.sum())
    else:
        num_unlabeled_certain = 0
    time_remaining_s = estimated_duration_s - status_duration
    return {
        "status": g_state.g_quick_status,
        "linear_status_epoch": g_state.linear_status_epoch,
        "trained_for_version": g_state.trained_for_version,
        "num_unlabeled_certain": num_unlabeled_certain,
        "num_total": len(g_state.features) if g_state.features is not None else "?",
        "time_remaining_s": f"{time_remaining_s:.2f}",
        "duration_std_s": f"{duration_std_s:.2f}",
        "duration_s": f"{estimated_duration_s:.2f}",
        "duration_perc": f"{time_remaining_s / estimated_duration_s if estimated_duration_s > 0 else 0:.2f}",
        "package_version": g_version,
        "num_labeled": len(g_state.labeled_indices) if g_state.labeled_indices else 0,
        # "performance": json.load(open(performance_path))
    }


def standard_json_response(error_code, error_message, http_status_code):
    """

    :param error_code: str
    :param error_message: str
    :param http_status_code: str
    :return: json response with compiled message
    """
    if error_message is None:
        error_message = " ".join(error_code.split("_")).capitalize()
    logger.info("status: {}, error message: {}".format(http_status_code, error_message))
    response = make_response(
        flask.jsonify({"error": {"code": error_code, "message": error_message}}),
        http_status_code,
    )
    return response


@app.errorhandler(500)
def general_server_error(e):
    """
    General server error response
    :param e: error
    :return: json response
    """
    logger.debug(e)
    return standard_json_response("general_server_error", None, 500)


def ui_script_entry():
    os.environ["PROJECT_ROOT"] = (
        os.path.expanduser(sys.argv[1])
        if os.getenv("PROJECT_ROOT") is None
        else os.getenv("PROJECT_ROOT")
    )
    _init()
    app.run(
        debug=str2bool(os.getenv("APP_DEBUG", False)),
        host="0.0.0.0",
        threaded=True,
        port=int(os.getenv("PORT", "8006")),
    )


if __name__ == "__main__":
    ui_script_entry()
