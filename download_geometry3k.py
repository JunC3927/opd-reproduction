import os
from datasets import load_dataset

# 数据保存目录
save_dir = "/raid/lwz/cj/datasets/geometry3k"

# Hugging Face 缓存目录，也放到 /raid/lwz/cj 下
cache_dir = "/raid/lwz/cj/hf_cache"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 如果服务器连 Hugging Face 慢，可以打开下面这一行
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ds = load_dataset(
    "hiyouga/geometry3k",
    cache_dir=cache_dir
)

print(ds)

sample = ds["train"][0]
print(sample.keys())
print("problem:", sample["problem"])
print("answer:", sample["answer"])
print("images:", sample["images"])

ds.save_to_disk(save_dir)

print(f"数据集已经保存到: {save_dir}")
