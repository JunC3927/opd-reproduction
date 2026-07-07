import argparse
import base64
import io

from datasets import load_from_disk
from openai import OpenAI


def image_to_base64_url(image):
    """
    把 PIL 图片转成 OpenAI image_url 可用的 base64 data URL。
    """
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{image_b64}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/raid/lwz/cj/datasets/geometry3k")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="qwen3vl-full-lm-epoch1")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    client = OpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
        timeout=3600,
    )

    ds = load_from_disk(args.data_path)[args.split]
    ex = ds[args.idx]

    image = ex["images"][0]
    raw_question = ex["problem"].replace("<image>", "").strip()
    gold = str(ex["answer"])

    image_url = image_to_base64_url(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": raw_question,
                },
            ],
        }
    ]

    print("=" * 100)
    print("idx:", args.idx)
    print("question:", raw_question)
    print("gold:", gold)
    print("=" * 100)

    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
    )

    pred = response.choices[0].message.content
    if pred is None:
        pred = ""

    pred = pred.strip()

    print("pred:")
    print(pred)
    print("=" * 100)


if __name__ == "__main__":
    main()
