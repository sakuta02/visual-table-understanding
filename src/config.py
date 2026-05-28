from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunConfig:
    gt_json_path: Path = PROJECT_ROOT / "gt.json"
    tables_path: Path = PROJECT_ROOT / "tables"
    output_path: Path = PROJECT_ROOT / "qwen_my_run"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_name: str = "Qwen"
    n_per_style: int = 50
    max_new_tokens: int = 2048
    max_image_side: int = 900

    @property
    def predictions_path(self) -> Path:
        return self.output_path / "predictions"

    @property
    def scores_path(self) -> Path:
        return self.output_path / "scores"

    @property
    def resized_path(self) -> Path:
        return self.output_path / "resized_images"

    def make_output_dirs(self) -> None:
        self.predictions_path.mkdir(parents=True, exist_ok=True)
        self.scores_path.mkdir(parents=True, exist_ok=True)
        self.resized_path.mkdir(parents=True, exist_ok=True)
