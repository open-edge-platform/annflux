# [AnnFlux] Annotation flux
## A research tool for exploring and annotating large datasets with Active Learning 

This standalone tool provides a basic interface for interacting with large datasets so that they can be explored and annotated efficiently. 

The extensible design of the tool allows researchers from in- and outside Intel to contribute to the development of the functionality

# License 

Apache 2.0, see [LICENSE.md](LICENSE)

# Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md)

# Install

1. Create and activate a new Conda environment:

   ```bash
   conda create -n AnnFlux_dev python=3.11
   conda activate AnnFlux_dev
   ```

2. Install the requirements:

   ```bash
   pip install -e .
   ```
   
# Init project, compute features and embed

`annflux go {PROJECT_FOLDER}`

`PROJECT_FOLDER` should have at least a `images` folder 

# Start basic UI

On commandline

`basic_ui {PROJECT_FOLDER}`

# Example

Use an image dataset with a folder of images with a .jpg extension. A good size is 5,000 to 10,000 images.

Extract to `~/annflux/data/envdataset/images`

On commandline

```bash
export HUGGINGFACE_CLIP_NAME={a hugging face CLIP model that supports the peft package}
```

```bash
annflux go ~/annflux/data/envdataset --start_labels Your_label_A Your_label_B Your_label_C`
```
Then

```bash
basic_ui ~/annflux/data/envdataset
```

Label some images, then

```bash
annflux train_then_features ~/annflux/data/envdataset
```

to perform parameter efficient fine-tuning of the (default) CLIP model, followed by computation of the adapted features.

# Tests and code coverage

```bash
export HUGGINGFACE_CLIP_NAME={a hugging face CLIP model that supports the peft package}
```

```bash
export USER_DATASET_PATH={your project folder with an 'images' folder inside}
```

```bash
python run_coverage.py
```


# Known issues

HIGH 
- (None)

MEDIUM
- [MEDIUM-1] Labels cannot be removed or edited in the UI. _Workaround_: edit `label_defs.json` in the project folder and restart UI
- [MEDIUM-2] Predicted labels with prob>0.50 are not shown for labeled data. _Workaround_: label through the top bar

# Changelog

## 1.0.0.0

- Final for open source
