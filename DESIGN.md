# Design philosophy

AnnFlux is set up as easily extendable research tool and functional components can be easily modified or replaced. 

To help this, communication between components is done as much as possible through common data formats (NumPy arrays, Pandas Dataframes) or common file formats (CSV, parquet).

If code becomes very complicated in its internal communication, or its execution becomes slow, this is a sign that a redesign or refactoring is needed.

# Workflow

## Main
1. Compute features for every image in a dataset
2. Embed features
3. Annotate the feature space through exploration and active learning
4. Train a new model

## Practical

1. Initialize the project
2. Main workflow
3. Export the model and run it as a service

# Components

NB: the files repo_results_to_embedding.py and train_indeed_image.py function as plumbing around feature embedding and model training respectively.

# Algorithms

- basic_ml.py: basic machine learning functions, e.g. softmax
- embeddings.py: compute embeddings from features
- feature_reconstruction_error.py: computes feature reconstruction error active learning measure
- most_needed.py: computes which samples should be annotated first to cover the feature space as quickly as possible

# Performance

- basic.py: computes basic performance measure + functionality for I/O

# Repository
Objects for persistence and provenance of ML objects

- dataset.py: represents a dataset (an immutable selection of data)
- model.py: represents a (trained) model
- resultset.py: the results of model M on dataset D
- repository.py: a collection of datasets, models, and resultsets

# Scripts
Provides tooling for the user to interact with AnnFlux

- annflux_cli.py: the main command-line interface (CLI) entry point
- run_tests.py: runs the tests

# Tests

- test_cli.py: tests CLI functionality
- test_train.py: tests separately the training functionality

# Tools
General functionality

- core.py: functionality to keep the state of the annotation process
- data.py: helper functionality to work with AnnFlux specific data
- io.py: helper functionality to work with data on disk
- mixed.py: general functionality, including logging

# Training
## AnnFlux

- clip.py: computes features using CLIP based architectures, also provides functionality for parameter efficient fine-tuning
- clip_server.py: functionality for running a trained CLIP mode as a webservice
- clip_shared.py: shared functionality between clip.py and clip_server.py
- feature_extractor.py: base class for general feature extractors
- quick.py: quick retraining of model

## Tensorflow

- tf_backend.py: linear models







