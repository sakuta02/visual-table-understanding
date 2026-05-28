import json
from pathlib import Path


def load_gt(gt_json_path: Path):
    with open(gt_json_path, "r", encoding="utf-8") as gt_file:
        return json.load(gt_file)


def is_complex_table(html):
    return "rowspan" in html["html"] or "colspan" in html["html"]


def get_style_names(tables_path: Path):
    return sorted(folder.stem for folder in tables_path.iterdir() if folder.is_dir())


def collect_images_by_style(tables_path: Path, gt_dict, n_per_style: int):
    images_by_style = {}

    for style in get_style_names(tables_path):
        style_images = sorted((tables_path / style / "rendered_pics").iterdir())

        simple_images = []
        complex_images = []

        for image_path in style_images:
            filename = image_path.stem
            gt_key = filename + ".png"

            if gt_key not in gt_dict:
                continue

            if is_complex_table(gt_dict[gt_key]):
                complex_images.append(image_path)
            else:
                simple_images.append(image_path)

        style_images = simple_images[: n_per_style // 2] + complex_images[: n_per_style // 2]
        images_by_style[style] = style_images

    return images_by_style
