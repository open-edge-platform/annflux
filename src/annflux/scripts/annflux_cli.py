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
from __future__ import annotations

import argparse
import os

from annflux.repo_results_to_embedding import embed_and_prepare
from annflux.shared import AnnfluxSource
from annflux.train_indeed_image import init_folder, train_then_features
from annflux.training.annflux.feature_extractor import TrainParameters




def execute(arg_list: list[str] | None = None):
    # Create the main parser
    parser = argparse.ArgumentParser(description="AnnFlux command")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize AnnFlux")
    init_parser.add_argument("folder", type=str, help="Data folder")
    data_csv_path_help = "The filename of images CSV"
    init_parser.add_argument(
        "--data_csv_path",
        type=str,
        help=data_csv_path_help,
        default="images.csv",
    )
    label_column_name_help = "The name for the true label if externally provided"
    init_parser.add_argument(
        "--label_column_name",
        type=str,
        help=label_column_name_help,
        default="label",
    )
    start_labels_help = "Annotation labels"

    init_parser.add_argument(
        "--start_labels", nargs="+", type=str, help=start_labels_help, default=[]
    )
    #
    features_parser = subparsers.add_parser(
        "train_then_features",
        help="Train model if labels are available, then compute features",
    )
    features_parser.add_argument("folder", type=str, help="Data folder")
    architecture_help = "Neural architecture to use to compute features. NB not all backends support all architectures"
    default_architecture = "clip"
    features_parser.add_argument(
        "--architecture",
        type=str,
        help=architecture_help,
        default="clip",
        choices=["clip", default_architecture],
    )
    features_parser.add_argument(
        "--train_peft",
        type=str,
        help="TODO",
        default="train_peft",
        choices=["train_peft", "train_deep"],
    )
    features_parser.add_argument("--num_epochs", type=int, help="TODO", default=15)
    backend_help = "Backend used for training models."
    features_parser.add_argument(
        "--backend",
        type=str,
        help=backend_help,
        choices=["otx", "naturalis-ai", "annflux"],
        default="annflux",
    )

    embedding_parser = subparsers.add_parser(
        "embed", help="Embed the features in the 2D space for display"
    )
    embedding_parser.add_argument("folder", type=str, help="Data folder")

    go_parser = subparsers.add_parser(
        "go", help="Init project, compute features, and embed in one go"
    )
    go_parser.add_argument("folder", type=str, help="Data folder")
    go_parser.add_argument(
        "--data_csv_path", type=str, help=data_csv_path_help, default="images.csv"
    )
    go_parser.add_argument(
        "--label_column_name", type=str, help=label_column_name_help, default="label"
    )
    go_parser.add_argument(
        "--start_labels", nargs="+", type=str, help=start_labels_help, default=[]
    )
    go_parser.add_argument(
        "--exclusivity", nargs="+", type=str, help="Exclusivity", default=[]
    )
    go_parser.add_argument(
        "--architecture",
        type=str,
        help=architecture_help,
        default=default_architecture,
        choices=[
            "clip",
        ],
    )
    #
    export_parser = subparsers.add_parser("export", help="TODO")
    export_parser.add_argument("folder", type=str, help="Data folder")
    export_parser.add_argument("out_folder", type=str, help="Model package folder")

    args = parser.parse_args(arg_list)
    folder = os.path.expanduser(args.folder)
    if args.command == "init":
        source = init_folder(
            AnnfluxSource(folder),
            label_column_name=args.label_column_name,
            start_labels=args.start_labels,
        )
        print(f"Initialized AnnFlux in folder {source.working_folder}")
    elif args.command == "train_then_features":
        train_then_features(
            AnnfluxSource(folder),
            architecture=args.architecture,
            train_method=args.train_peft,
            train_parameters=TrainParameters(num_epochs=args.num_epochs),
        )
    elif args.command == "embed":
        embed_and_prepare(AnnfluxSource(folder))
    elif args.command == "go":
        exclusivity = args.exclusivity
        go_command(
            folder,
            args.start_labels,
            exclusivity,
            args.architecture,
            args.label_column_name,
        )


def go_command(
    folder, start_labels, exclusivity=None, architecture="clip", label_column_name=None
):
    if exclusivity is None:
        exclusivity = []
    source = init_folder(
        AnnfluxSource(folder),
        label_column_name=label_column_name,
        start_labels=start_labels,
        exclusivity_groups=[group_.split(",") for group_ in exclusivity],
    )
    print(f"Initialized AnnFlux in folder {source.working_folder}")
    train_then_features(AnnfluxSource(folder), architecture=architecture)
    embed_and_prepare(AnnfluxSource(folder))


if __name__ == "__main__":
    execute()
