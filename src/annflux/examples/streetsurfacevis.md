
```bash
# see https://zenodo.org/records/11449977
mkdir ~/annflux/data/streetSurfaceVis
cd ~/annflux/data/streetSurfaceVis
curl https://zenodo.org/records/11449977/files/s_256.zip?download=1 > streetSurfaceVis.zip
unzip -j streetSurfaceVis.zip -d "images"
```

```bash
python ../scripts/annflux_cli.py go ~/annflux/data/streetSurfaceVis --start_labels Asphalt Concrete Paving_stones Sett Unpaved --exclusivity Asphalt,Concrete,Paving_stones,Sett,Unpaved 
```

```bash
export APP_DEBUG=1; python ../ui/basic/run_server.py ~/annflux/data/streetSurfaceVis 
```

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
annflux_folder = "~/annflux/data/streetSurfaceVis/annflux"
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

```bash
python ../scripts/annflux_cli.py train_then_features ~/annflux/data/streetSurfaceVis 
```