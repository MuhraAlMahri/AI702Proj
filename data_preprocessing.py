import json
import random
from datasets import load_dataset

# 加载 Hugging Face 数据集
# 需要指定配置名称 'language-language-3'
dataset = load_dataset(
    "SCI-Benchmark/self-contradictory",
    name="language-language-3",  # 指定配置名称
    split="full"
)

# 解析数据格式
formatted_data = [
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": f'{item["context"]}',
        "output": f'The instructions "{item["instruction1"]}" and "{item["instruction2"]}" are contradictory. Please clarify your preference.'
    }
    for item in dataset
]

# 打乱数据顺序（确保训练/测试集均匀）
random.shuffle(formatted_data)

# 计算数据划分索引
total_samples = len(formatted_data)
train_size = int(0.8 * total_samples)  # 80% 作为训练集

# 划分训练、测试集
train_data = formatted_data[:train_size]
test_data = formatted_data[train_size:]

# 保存为单个 JSON 文件，但使用列表格式
all_data = {
    "train": [
        {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        }
        for item in train_data
    ],
    "test": [
        {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        }
        for item in test_data
    ]
}

# 保存数据
with open("contradiction_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)

print(f"Data saved to contradiction_data.json")
print(f"Training set: {len(train_data)} samples")
print(f"Test set: {len(test_data)} samples")
