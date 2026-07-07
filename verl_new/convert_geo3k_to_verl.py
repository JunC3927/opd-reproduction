from datasets import load_from_disk

src = "/raid/lwz/cj/datasets/geometry3k"
out_dir = "/home/user01/data/geo3k"

ds = load_from_disk(src)

def convert(example, idx):
    problem = example["problem"]
    answer = str(example["answer"])

    return {
        "data_source": "geo3k",
        "prompt": [
            {
                "role": "user",
                "content": problem,
            }
        ],
        "images": example["images"],
        "ability": "geometry",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": idx,
            "answer": answer,
            "problem": problem,
        },
    }

train = ds["train"].map(convert, with_indices=True)
val = ds["validation"].map(convert, with_indices=True)

train.to_parquet(f"{out_dir}/train.parquet")
val.to_parquet(f"{out_dir}/test.parquet")

print(train)
print(val)
print("saved to:", out_dir)
print("sample:", train[0])
