import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")

    parser.add_argument("--data_path", type=str, default="/raid/lwz/cj/datasets/geometry3k")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_eval", type=int, default=300)

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retry", type=int, default=2)
    return parser.parse_args()


def image_to_base64_url(image):
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def normalize_answer(x):
    x = str(x).strip().lower()
    x = x.replace("\\boxed", "")
    x = x.replace("{", "").replace("}", "")
    x = x.replace("$", "")
    x = x.replace("°", "")
    x = re.sub(r"\s+", "", x)
    return x


def extract_answer(text):
    text = str(text)

    # 优先抽 \boxed{...}
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # 再抽 Final answer: ...
    final = re.findall(r"final answer\s*[:：]\s*([^\n]+)", text, flags=re.I)
    if final:
        ans = final[-1].strip()
        ans = ans.replace("\\boxed", "").replace("{", "").replace("}", "")
        return ans.strip()

    # 最后兜底：抽最后一个数字
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    if nums:
        return nums[-1].strip()

    return ""


def load_records(path):
    records = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                records[int(item["idx"])] = item
            except Exception:
                pass
    return records


def save_records(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for idx in sorted(records):
            f.write(json.dumps(records[idx], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_messages(ex):
    image = ex["images"][0]
    raw_question = ex["problem"].replace("<image>", "").strip()

    prompt = f"""{raw_question}

Look at the image and solve the geometry problem.

Give a concise solution. Do not repeat the question.
At the end, write the final answer in exactly this format:
Final answer: \\boxed{{your answer}}"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(image)},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return raw_question, prompt, messages


def run_one(args, idx, ex):
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=3600,
    )

    raw_question, prompt, messages = build_messages(ex)
    gold = str(ex["answer"])

    last_error = None

    for _ in range(args.retry + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=args.max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )

            choice = resp.choices[0]
            response_text = (choice.message.content or "").strip()
            finish_reason = choice.finish_reason

            completion_tokens = None
            if getattr(resp, "usage", None) is not None:
                completion_tokens = getattr(resp.usage, "completion_tokens", None)

            hit_max_tokens = (
                finish_reason == "length"
                or (
                    completion_tokens is not None
                    and completion_tokens >= args.max_tokens
                )
            )

            pred_answer = extract_answer(response_text)
            raw_correct = normalize_answer(pred_answer) == normalize_answer(gold)

            # 严格准确率：只要达到 max_tokens，就直接判错
            strict_correct = raw_correct and not hit_max_tokens

            return {
                "idx": idx,
                "raw_question": raw_question,
                "prompt": prompt,
                "gold": gold,
                "pred_answer": pred_answer,
                "response": response_text,

                "raw_correct": bool(raw_correct),
                "strict_correct": bool(strict_correct),

                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "max_tokens": args.max_tokens,
                "hit_max_tokens": bool(hit_max_tokens),

                "manual_correct": None,
            }

        except Exception as e:
            last_error = repr(e)
            time.sleep(2)

    return {
        "idx": idx,
        "raw_question": raw_question,
        "prompt": prompt,
        "gold": gold,
        "pred_answer": "",
        "response": "",

        "raw_correct": False,
        "strict_correct": False,

        "finish_reason": None,
        "completion_tokens": None,
        "max_tokens": args.max_tokens,
        "hit_max_tokens": False,

        "manual_correct": None,
        "error": last_error,
    }


def main():
    args = parse_args()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_path)[args.split]
    num_eval = min(args.num_eval, len(ds))

    records = load_records(args.output_path)
    todo = [
        i for i in range(num_eval)
        if i not in records or not records[i].get("response", "")
    ]

    print("=" * 80)
    print("Geometry3K vLLM strict eval")
    print("model:", args.model_name)
    print("split:", args.split)
    print("num_eval:", num_eval)
    print("todo:", len(todo))
    print("max_tokens:", args.max_tokens)
    print("output:", args.output_path)
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one, args, idx, ds[idx]): idx
            for idx in todo
        }

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating")

        for fut in pbar:
            item = fut.result()
            records[item["idx"]] = item

            done = [records[i] for i in range(num_eval) if i in records]
            raw_correct = sum(x.get("raw_correct", False) for x in done)
            strict_correct = sum(x.get("strict_correct", False) for x in done)
            hit_max = sum(x.get("hit_max_tokens", False) for x in done)
            errors = sum(1 for x in done if x.get("error"))

            pbar.set_postfix(
                done=f"{len(done)}/{num_eval}",
                raw=f"{raw_correct / max(len(done), 1):.2%}",
                strict=f"{strict_correct / max(len(done), 1):.2%}",
                hit_max=f"{hit_max / max(len(done), 1):.2%}",
                errors=errors,
            )

            save_records(args.output_path, records)

    final = [records[i] for i in range(num_eval) if i in records]
    raw_correct = sum(x.get("raw_correct", False) for x in final)
    strict_correct = sum(x.get("strict_correct", False) for x in final)
    hit_max = sum(x.get("hit_max_tokens", False) for x in final)
    errors = sum(1 for x in final if x.get("error"))

    print("=" * 80)
    print("Finished")
    print("output:", args.output_path)
    print(f"records: {len(final)}/{num_eval}")
    print(f"errors: {errors}")
    print(f"raw acc: {raw_correct}/{num_eval} = {raw_correct / num_eval:.2%}")
    print(f"strict acc: {strict_correct}/{num_eval} = {strict_correct / num_eval:.2%}")
    print(f"hit max tokens: {hit_max}/{num_eval} = {hit_max / num_eval:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()