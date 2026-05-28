from src.config import RunConfig
from src.pipeline import evaluate_predictions, load_run_data, run_inference


def main():
    config = RunConfig(
        n_per_style=50,
        max_image_side=900,
        max_new_tokens=2048,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    config.make_output_dirs()

    gt_dict, style_names, images_by_style = load_run_data(config)

    time_by_style = run_inference(config, style_names, images_by_style)
    evaluate_predictions(config, gt_dict, style_names, time_by_style)


if __name__ == "__main__":
    main()
