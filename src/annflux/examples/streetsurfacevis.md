Use the public (Creative Commons Attribution Share Alike 4.0 International) road type dataset 'StreetSurfaceVis' (https://zenodo.org/records/11449977 / https://www.nature.com/articles/s41597-024-04295-9).

The dataset contains 9122 images of various road types.

This example assumes you have created the "annflux" directory in your home directory, but it works for any base directory (adjust the paths in the commands below).

Download the data
```bash
# see https://zenodo.org/records/11449977
mkdir -p ~/annflux/data/streetSurfaceVis
cd ~/annflux/data/streetSurfaceVis
curl --location --output streetSurfaceVis.zip "https://zenodo.org/records/11449977/files/s_256.zip?download=1"
unzip -j streetSurfaceVis.zip -d "images"
# the original zip unfortunately contains two 0-bytes files
find ~/annflux/data/streetSurfaceVis/images -size 0 -delete
```
ANNFLUX GO: initialize the project, compute features and embed them
```bash
cd ..
export PYTHONPATH="..:$PYTHONPATH"
python scripts/annflux_cli.py go ~/annflux/data/streetSurfaceVis --start_labels Asphalt Concrete Paving_stones Sett Unpaved --exclusivity Asphalt,Concrete,Paving_stones,Sett,Unpaved 
```
Run UI
```bash
cd ..
export PYTHONPATH="..:$PYTHONPATH"
export APP_DEBUG=1; python ui/basic/run_server.py ~/annflux/data/streetSurfaceVis 
```
## Optional: set the true labels
```bash
cd ~/annflux/data/streetSurfaceVis
curl https://zenodo.org/records/11449977/files/streetSurfaceVis_v1_0.csv?download=1 > streetSurfaceVis_v1_0.csv
```

```python
# run in examples dir
import os
import pandas
import json
os.chdir(os.path.expanduser("~/annflux/data/streetSurfaceVis"))
t = pandas.read_csv("streetSurfaceVis_v1_0.csv", dtype={"mapillary_image_id": str})
annflux_folder = os.path.expanduser("~/annflux/data/streetSurfaceVis/annflux")
annotations_path = os.path.join(annflux_folder, "labels.json")
if os.path.exists(annotations_path):
    with open(annotations_path) as f:
        annotations = json.load(f)
else:
    annotations = {}
test_only = True # if True will only set the true labels for the test set
test_uids = set(t.mapillary_image_id)
if test_only:
    with open(os.path.join(annflux_folder, "split.json")) as f:
        test_uids = set(json.load(f)["test"])
print(f"{len(test_uids)=}")
for _, row in t.iterrows():
    uid = str(row.mapillary_image_id)
    if uid in test_uids:
        annotations[uid] = row.surface_type.capitalize()
with open(os.path.join(annflux_folder, "labels.json"), "w") as f:
    json.dump(annotations, f, indent=2)
```

## Train
```bash
python ../scripts/annflux_cli.py train_then_features ~/annflux/data/streetSurfaceVis 
```

## Export
