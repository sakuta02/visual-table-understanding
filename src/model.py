from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.html_utils import wrap_html


PROMPT = """Convert the table in the image into an HTML table. Follow these rules strictly: - Return only raw HTML. - Do not wrap the answer in Markdown code fences. - Do not add any explanation. - Use only these tags: <table>, <tr>, <th>, <td>. - Preserve the visual table structure exactly. - Correctly represent merged cells using rowspan and colspan. - Use <th> for all header cells, including multi-level headers. - Use <td> for body cells. - Preserve the reading order from top to bottom and left to right. - Keep the original text content exactly as shown. - Do not normalize, translate, summarize, or infer text. - If a cell is empty, output an empty cell tag. - Do not add CSS, style attributes, class attributes, id attributes, <thead>, <tbody>, or <caption>. The output must be a single valid HTML <table>...</table>."""


class TableRecognitionModel:
    def __init__(self, model_id: str, max_new_tokens: int):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def run(self, image_path):
        image_path = Path(image_path).resolve()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return wrap_html(output_text)


def resize_image(image_path, style, resized_path: Path, max_image_side: int):
    image_path = Path(image_path)

    style_resized_path = resized_path / style
    style_resized_path.mkdir(parents=True, exist_ok=True)

    out_path = style_resized_path / image_path.name

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    scale = min(max_image_side / max(w, h), 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    image.save(out_path)

    return out_path
