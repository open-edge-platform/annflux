![AnnFlux Logo]([src/annflux/ui/basic/static/annflux.png](https://github.com/open-edge-platform/annflux/blob/main/src/annflux/ui/basic/static/annflux.png))

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
# Examples

## StreetSurfaceVis

The public (Creative Commons Attribution Share Alike 4.0 International) road type dataset 'StreetSurfaceVis' (https://zenodo.org/records/11449977 / https://www.nature.com/articles/s41597-024-04295-9).

See [StreetSurfaceVis](src/annflux/examples/streetsurfacevis.md)

## From images folder    
## Init project, compute features and embed

`annflux go {PROJECT_FOLDER}`

`PROJECT_FOLDER` should have at least a `images` folder

## Example

Use an image dataset with a folder of images with a .jpg extension. A good size is 5,000 to 10,000 images.

Extract to `~/annflux/data/envdataset/images`

On commandline

```bash
export HUGGINGFACE_CLIP_NAME={a hugging face CLIP model that supports the peft package}
```

```bash
annflux go ~/annflux/data/envdataset --start_labels Your_label_A Your_label_B Your_label_C`
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

# Changelog

## 1.0.1.0

- Basic UI contribution from Naturalis

## 1.0.0.0

- Final first release

## 0.9.3.0

- Improved test reporting

## 0.9.3.0

- Improved test reporting

## 0.9.2.0

- Preparing for open source
