import glob
import math
import os
from pathlib import Path

import PIL
import pandas
from dateutil.parser import ParserError, parse
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from annflux.tools.io import basename_no_extension


def tile_and_save_image_with_padding(
    img_path,
    output_dir,
    patch_size,
    overlap=0,
    pad_color=(0, 0, 0),
    prefix="",
    skip_existing: bool | int = False,
    original_label=None,
):
    """
    Split an image into square patches with optional overlap and padding, then save them.

    Parameters:
    - img_path: path to input image
    - output_dir: where to save patches
    - patch_size: width/height of square patch
    - overlap: overlapping pixels between patches
    - pad_color: RGB color tuple for padding (default black)
    """
    basename = os.path.splitext(os.path.basename(img_path))[0]

    PIL.Image.MAX_IMAGE_PIXELS = 259341992 * 2  # for things such as orthophotos
    img = Image.open(img_path)
    width, height = img.size
    step = patch_size - overlap

    # Calculate number of patches needed
    num_cols = math.ceil((width - overlap) / step)
    num_rows = math.ceil((height - overlap) / step)

    # Calculate required padded dimensions
    pad_width = (num_cols - 1) * step + patch_size
    pad_height = (num_rows - 1) * step + patch_size

    pad_right = pad_width - width
    pad_bottom = pad_height - height

    # Apply padding if needed
    if (pad_right > 0 or pad_bottom > 0) and skip_existing < 2:
        img = ImageOps.expand(img, (0, 0, pad_right, pad_bottom), fill=pad_color)

    os.makedirs(output_dir, exist_ok=True)

    if len(prefix) > 0:
        if "$" in prefix:  # TODO: other options
            prefix = os.path.split(os.path.dirname(img_path))[-1] + "_"

    patch_tuples = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * step
            y = row * step
            filename = f"{prefix}{basename}_x{x}_y{y}.jpg"
            patch_basename = f"{prefix}{basename}_x{x}_y{y}"
            try:
                datetime_ = parse(basename)
            except ParserError:
                datetime_ = None
            patch_path = os.path.join(output_dir, filename.replace("-", "_"))
            if not os.path.exists(patch_path) or not skip_existing:
                box = (x, y, x + patch_size, y + patch_size)
                patch = img.crop(box)
                patch.save(patch_path)
            patch_tuples.append(
                (
                    img_path,
                    basename,
                    patch_basename,
                    x,
                    y,
                    datetime_.isoformat() if datetime_ is not None else "n/a",
                    "0,1",
                    patch_path,
                    original_label,
                )
            )
            # print(f"Saved {filename}")
    return patch_tuples


def tile_and_save(
    output_folder="patches",
    file_paths=None,
    prefix="",
    skip_existing=1,
    patch_size=512,
    original_label: dict[str, str] = None,
) -> pandas.DataFrame:
    if file_paths is None:
        file_paths = []
    if original_label is None:
        original_label = {}
    patch_tuples = []
    for fn in tqdm(file_paths, desc="tiling images", total=len(file_paths)):
        if os.path.isdir(fn):
            continue
        try:
            patch_tuples.extend(
                tile_and_save_image_with_padding(
                    img_path=fn,
                    output_dir=output_folder,
                    patch_size=patch_size,
                    overlap=0,
                    pad_color=(0, 0, 0),  # Black padding
                    prefix=prefix,
                    skip_existing=skip_existing,
                    original_label=original_label.get(basename_no_extension(fn)),
                )
            )
        except UnidentifiedImageError:
            # TODO(improvement): consider moving the (corrupted) file to another folder
            print(f"Failed to read {fn}")
    result = pandas.DataFrame(
        data=patch_tuples,
        columns=(
            "original_image_path",
            "original_image_id",
            "image_id",
            "patch_x",
            "patch_y",
            "datetime",
            "label",
            "patch_path",
            "label_original",
        ),
    )
    result["record_id"] = result.original_image_id
    return result

def execute(
    folder,
    original_data_path: str | None = None,  # Path to CSV file containing metadata
    output_folder: str = None,  # Folder where patches will be saved
    patch_size: int = 512,
) -> pandas.DataFrame:
    if output_folder is None:
        output_folder = os.path.join(folder, "images")
    if original_data_path is not None and os.path.exists(original_data_path):
        t = pandas.read_csv(original_data_path)
        t["filename"] = t.image_id.apply(lambda x_: f"NMR_NS_{x_.split(':')[1]}")
        original_label: dict[str, str] = dict(
            zip(t.filename, t.taxon_full_name.apply(lambda x_: x_.split(" (")[0]))
        )
    else:
        original_label: dict[str, str] = {}

    file_paths: list[str] = glob.glob(f"{folder}/original/*.jpg")
    table = tile_and_save(
        output_folder=output_folder,
        file_paths=file_paths,
        patch_size=patch_size,
        original_label=original_label,
        skip_existing=2,
    )
    table.to_csv(Path(output_folder) / ".." / "images.csv", index=False)

    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tile and save images")
    parser.add_argument(
        "folder",
        help="Path to input folder",
    )
    parser.add_argument(
        "--output_folder", help="Output folder for patches", default=None
    )
    parser.add_argument(
        "--original_data_path",
        help="Path to original data CSV",
        default=None,
    )
    parser.add_argument(
        "--patch_size",
        help="TODO",
        default=512,
        type=int
    )
    args = parser.parse_args()

    execute(args.folder, args.output_folder, patch_size=args.patch_size)
