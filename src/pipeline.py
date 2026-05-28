import gc
import json
import time

from src.data import collect_images_by_style, get_style_names, load_gt


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_run_data(config):
    gt_dict = load_gt(config.gt_json_path)
    style_names = get_style_names(config.tables_path)
    images_by_style = collect_images_by_style(
        config.tables_path,
        gt_dict,
        config.n_per_style,
    )
    return gt_dict, style_names, images_by_style


def run_inference(config, style_names, images_by_style):
    import torch
    from tqdm import tqdm

    from src.model import TableRecognitionModel, resize_image

    model = TableRecognitionModel(config.model_id, config.max_new_tokens)
    time_by_style = {}

    for style in style_names:
        pred_path = (
            config.predictions_path
            / f"qwen_{style}_{config.max_image_side}_{config.max_new_tokens}.json"
        )

        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        else:
            predictions = {}

        print()
        print("=" * 80)
        print("Стиль:", style)
        print("Изображения:", len(images_by_style[style]))
        print("Уже предсказано:", len(predictions))
        print("=" * 80)

        processed_count = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        for image_path in tqdm(images_by_style[style]):
            filename = image_path.name

            if filename in predictions and predictions[filename]:
                continue

            resized_image_path = resize_image(
                image_path,
                style,
                config.resized_path,
                config.max_image_side,
            )
            pred_html = model.run(resized_image_path)

            predictions[filename] = pred_html
            processed_count += 1

            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.time() - start_time

        if processed_count > 0:
            avg_time = total_time / processed_count
        else:
            avg_time = 0

        time_by_style[style] = {
            "total_time": total_time,
            "avg_time": avg_time,
            "processed_count": processed_count,
            "total_images": len(images_by_style[style]),
        }

        print("Сохранено:", pred_path)
        print("Обработано новых изображений:", processed_count)
        print("Итоговое время:", round(total_time, 2), "секунд")
        print("Среднее время:", round(avg_time, 2), "секунд/изображение")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return time_by_style


def evaluate_predictions(config, gt_dict, style_names, time_by_style=None):
    import pandas as pd

    from src.html_utils import normalize_table_html
    from src.teds import TEDS

    if time_by_style is None:
        time_by_style = {}

    results = []

    for style in style_names:
        pred_path = (
            config.predictions_path
            / f"qwen_{style}_{config.max_image_side}_{config.max_new_tokens}.json"
        )

        if not pred_path.exists():
            print()
            print("Нет predictions файла, пропускаю:", pred_path)
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_dict = json.load(f)

        clean_pred_dict = {}

        for filename, pred_html in pred_dict.items():
            if pred_html is not None and str(pred_html).strip():
                clean_pred_dict[filename] = normalize_table_html(pred_html)

        true_dict = {}

        for filename in clean_pred_dict:
            if filename in gt_dict:
                true_dict[filename] = gt_dict[filename]

        true_eval_dict = {
            filename: {"html": normalize_table_html(gt_dict[filename]["html"])}
            for filename in true_dict
        }

        teds_full = TEDS(structure_only=False, ignore_nodes=["b"])
        teds_structure = TEDS(structure_only=True, ignore_nodes=["b"])

        full_scores = teds_full.batch_evaluate(clean_pred_dict, true_eval_dict)
        structure_scores = teds_structure.batch_evaluate(clean_pred_dict, true_eval_dict)

        matched_filenames = list(full_scores.keys())
        mean_full = mean(list(full_scores.values()))
        mean_structure = mean(list(structure_scores.values()))

        simple_full = []
        simple_structure = []
        complex_full = []
        complex_structure = []

        for filename in matched_filenames:
            table_type = true_dict[filename]["type"]

            if table_type == "simple":
                simple_full.append(full_scores[filename])
                simple_structure.append(structure_scores[filename])

            elif table_type == "complex":
                complex_full.append(full_scores[filename])
                complex_structure.append(structure_scores[filename])

        mean_simple_full = mean(simple_full)
        mean_simple_structure = mean(simple_structure)
        mean_complex_full = mean(complex_full)
        mean_complex_structure = mean(complex_structure)

        time_info = time_by_style.get(style, {})

        style_result = {
            "model": config.model_name,
            "style": style,
            "n": len(matched_filenames),
            "n_simple": len(simple_full),
            "n_complex": len(complex_full),
            "mean_teds_full": mean_full,
            "mean_teds_structure": mean_structure,
            "mean_simple_full": mean_simple_full,
            "mean_simple_structure": mean_simple_structure,
            "mean_complex_full": mean_complex_full,
            "mean_complex_structure": mean_complex_structure,
            "total_time": time_info.get("total_time"),
            "avg_time": time_info.get("avg_time"),
        }

        results.append(style_result)

        score_path = (
            config.scores_path
            / f"qwen_{style}_{config.max_image_side}_{config.max_new_tokens}_scores.json"
        )

        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **style_result,
                    "full_scores": full_scores,
                    "structure_scores": structure_scores,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(style)
        print("n:", len(matched_filenames))
        print("n_simple:", len(simple_full))
        print("n_complex:", len(complex_full))
        print("TEDS full:", round(mean_full, 4))
        print("TEDS structure:", round(mean_structure, 4))
        print("TEDS simple full:", round(mean_simple_full, 4))
        print("TEDS simple structure:", round(mean_simple_structure, 4))
        print("TEDS complex full:", round(mean_complex_full, 4))
        print("TEDS complex structure:", round(mean_complex_structure, 4))

    summary_df = pd.DataFrame(results)

    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(
            by="mean_teds_structure",
            ascending=False,
        )

    summary_csv_path = (
        config.output_path / f"qwen_summary_{config.max_image_side}_{config.max_new_tokens}.csv"
    )
    summary_json_path = (
        config.output_path / f"qwen_summary_{config.max_image_side}_{config.max_new_tokens}.json"
    )

    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return summary_df
