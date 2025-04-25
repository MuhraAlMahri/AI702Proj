import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # 导入 tqdm 库

# 加载微调后的 LLaMA
model_path = "./lora_llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 读取数据集
with open("test.json", "r") as f:
    test_data = json.load(f)

results = []
batch_size = 5  # 每次写入文件的条数

# 使用 tqdm 包装 test_data 以显示进度条
for idx, item in enumerate(tqdm(test_data, desc="Processing")):
    input_text = f"Input: {item['input']}\nOutput:"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.2, top_p=0.9)
    
    # 解码模型输出
    model_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 存储结果
    results.append({
        "input": item["input"],
        "expected_output": item["output"],  # 真实答案
        "model_output": model_output  # LLaMA 生成的答案
    })

    # 每5条写入一次文件
    if (idx + 1) % batch_size == 0:
        with open("inference_results_finetuned.json", "w") as f:
            json.dump(results, f, indent=4)

# 确保写入剩余的结果
if len(results) % batch_size != 0:
    with open("inference_results_finetuned.json", "w") as f:
        json.dump(results, f, indent=4)

print("✅ 推理完成，结果已保存到 inference_results_finetuned.json")