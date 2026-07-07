import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


VALID_LABELS = {"correct", "wrong", "uncertain"}


SYSTEM_PROMPT = """
你是一个严格但公平的数学答案裁判。

你的任务：
根据题目描述、标准答案和模型完整输出，判断模型最终答案是否正确。

判断规则：
1. 主要看模型完整输出中的最终答案是否与标准答案数学等价。
2. 不要求推理过程完全一致，只要最终答案正确即可。
3. 接受等价形式，例如：
   - 63 和 \\boxed{63}
   - 1/2 和 0.5
   - x = 63 和 63
   - 角度答案有无 ° 都可以
4. 如果模型最终答案错误，即使过程部分正确，也判 wrong。
5. 如果模型没有给出明确最终答案，或者无法判断，判 uncertain。
6. 不要重新解题，只判断模型输出是否匹配标准答案。

只返回 JSON，不要返回 markdown。

JSON 格式：
{
  "label": "correct" | "wrong" | "uncertain",
  "confidence": 0.0,
  "reason": "简短中文理由"
}
""".strip()


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    os.replace(tmp, path)


def extract_json(text):
    text = str(text).strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("无法解析 JSON: " + text[:300])


def build_prompt(item, max_response_chars=8000):
    question = str(item.get("raw_question", ""))
    gold = str(item.get("gold", ""))
    response = str(item.get("response", ""))

    if len(response) > max_response_chars:
        response = response[:max_response_chars] + "\n...[TRUNCATED]"

    return f"""
请判断下面这个几何题模型答案是否正确。

[题目描述]
{question}

[标准答案]
{gold}

[模型完整输出]
{response}

请只返回 JSON。
""".strip()


def mark_uncertain_by_max_token(item):
    item["deepseek_label"] = "uncertain"
    item["deepseek_correct"] = None
    item["deepseek_confidence"] = 1.0
    item["deepseek_reason"] = "模型输出达到 max_tokens，上下文被截断，直接判为 uncertain"
    item["deepseek_error"] = ""
    item["deepseek_skipped_by_hit_max_tokens"] = True
    return item


def judge_one(args, item):
    # 如果模型输出达到 max_tokens，直接 uncertain，不调用 DeepSeek
    if item.get("hit_max_tokens") is True:
        return mark_uncertain_by_max_token(item)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(item, args.max_response_chars)},
    ]

    last_error = None

    for attempt in range(args.retry + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_tokens,
                )

            content = resp.choices[0].message.content or ""
            obj = extract_json(content)

            label = str(obj.get("label", "uncertain")).lower().strip()
            if label not in VALID_LABELS:
                label = "uncertain"

            try:
                confidence = float(obj.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))

            if label == "correct":
                correct = True
            elif label == "wrong":
                correct = False
            else:
                correct = None

            item["deepseek_label"] = label
            item["deepseek_correct"] = correct
            item["deepseek_confidence"] = confidence
            item["deepseek_reason"] = str(obj.get("reason", "")).strip()
            item["deepseek_error"] = ""
            item["deepseek_skipped_by_hit_max_tokens"] = False

            return item

        except Exception as e:
            last_error = repr(e)
            time.sleep(1 + attempt)

    item["deepseek_label"] = "uncertain"
    item["deepseek_correct"] = None
    item["deepseek_confidence"] = 0.0
    item["deepseek_reason"] = "DeepSeek 调用失败，需人工检查"
    item["deepseek_error"] = last_error
    item["deepseek_skipped_by_hit_max_tokens"] = False
    return item


def already_done(item):
    return item.get("deepseek_label") in VALID_LABELS and not item.get("deepseek_error")


def print_stats(records):
    total = len(records)

    judged = [x for x in records if x.get("deepseek_label") in VALID_LABELS]

    correct = sum(1 for x in judged if x.get("deepseek_label") == "correct")
    wrong = sum(1 for x in judged if x.get("deepseek_label") == "wrong")
    uncertain = sum(1 for x in judged if x.get("deepseek_label") == "uncertain")
    errors = sum(1 for x in judged if x.get("deepseek_error"))
    hit_max_uncertain = sum(
        1 for x in judged
        if x.get("deepseek_skipped_by_hit_max_tokens") is True
    )

    acc_total = correct / total if total else 0.0

    print("\n========== DeepSeek Judge 统计 ==========")
    print(f"总数: {total}")
    print(f"已判断: {len(judged)}")
    print(f"correct: {correct}")
    print(f"wrong: {wrong}")
    print(f"uncertain: {uncertain}")
    print(f"其中 hit_max_tokens 直接 uncertain: {hit_max_uncertain}")
    print(f"errors: {errors}")
    print("----------------------------------------")
    print(f"准确率: {correct}/{total} = {acc_total:.2%}")
    print("========================================\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DEEPSEEK_API_KEY", ""),
    )

    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-response-chars", type=int, default=8000)
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    input_records = load_jsonl(args.input)

    if os.path.exists(args.output) and not args.overwrite:
        output_records = load_jsonl(args.output)
        if len(output_records) == len(input_records):
            records = output_records
            print("[INFO] 检测到已有输出文件，自动断点续跑:", args.output)
        else:
            records = input_records
            print("[WARN] 输出文件长度不一致，从输入文件重新开始")
    else:
        records = input_records

    todo = [
        i for i, x in enumerate(records)
        if args.overwrite or not already_done(x)
    ]

    print("=" * 80)
    print("DeepSeek Judge for Geo3K eval jsonl")
    print("input:", args.input)
    print("output:", args.output)
    print("model:", args.model)
    print("todo:", len(todo))
    print("concurrency:", args.concurrency)
    print("=" * 80)

    done_since_save = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(judge_one, args, records[i].copy()): i
            for i in todo
        }

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Judging")

        for fut in pbar:
            i = futures[fut]
            records[i] = fut.result()

            done_since_save += 1
            if done_since_save >= args.save_every:
                save_jsonl(records, args.output)
                done_since_save = 0

            judged = [x for x in records if x.get("deepseek_label") in VALID_LABELS]
            correct = sum(1 for x in judged if x.get("deepseek_label") == "correct")
            uncertain = sum(1 for x in judged if x.get("deepseek_label") == "uncertain")

            pbar.set_postfix(
                judged=len(judged),
                correct=correct,
                uncertain=uncertain,
                acc=f"{correct / max(len(records), 1):.2%}",
            )

    save_jsonl(records, args.output)
    print("\n[DONE] saved:", args.output)
    print_stats(records)


if __name__ == "__main__":
    main()