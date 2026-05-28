from json import load
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_JSON_PATH = PROJECT_ROOT / "gt.json"
STYLES_DIR = PROJECT_ROOT / "styles"
OUTPUT_DIR = PROJECT_ROOT / "tables"


def render_tables(html_paths, screenshots):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 3000, "height": 3000})

        for i, html_file in enumerate(html_paths):
            output_path = screenshots / f"{html_file.stem}.png"

            page.goto(html_file.as_uri())
            page.wait_for_selector("table")
            page.locator("table").screenshot(path=str(output_path.resolve()))

            if (i + 1) % 50 == 0:
                print(f"Отрендерено {i + 1}/{len(html_paths)}")

        browser.close()


def render_style(style_path: Path, loaded, limit=500):
    style_name = style_path.stem

    tables_folder = OUTPUT_DIR / style_name
    rendered_html = tables_folder / "rendered_html"
    rendered_pics = tables_folder / "rendered_pics"

    rendered_html.mkdir(parents=True, exist_ok=True)
    rendered_pics.mkdir(parents=True, exist_ok=True)

    html_paths = []

    for table in list(loaded.keys())[:limit]:
        row_html = loaded[table]["html"]

        html_file = (rendered_html / table).with_suffix(".html")

        row_html = (
            f'<html><head><meta charset="UTF-8">'
            f'<link rel="stylesheet" href="{style_path.resolve().as_uri()}">'
            f"</head>"
            + row_html.replace("<html>", "", 1)
        )

        html_file.write_text(row_html, encoding="utf-8")
        html_paths.append(html_file.resolve())

    print(f"Рендерится стиль: {style_name}")
    render_tables(html_paths, rendered_pics)


with open(TABLES_JSON_PATH.resolve(), "r", encoding="utf-8") as f:
    loaded = load(f)

for style_file in sorted(STYLES_DIR.glob("*.css")):
    render_style(style_file, loaded, limit=500)

