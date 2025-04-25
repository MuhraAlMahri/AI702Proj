import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # 进度条库

# 📌 指定 LLaMA-3.2-1B-Instruct 模型
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = "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"

# 📌 加载 tokenizer 和模型（确保你有 Hugging Face 访问权限）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", use_auth_token=HF_TOKEN)

# 📌 读取数据集
with open("test.json", "r") as f:
    test_data = json.load(f)

results = []
batch_size = 5  # 每5条写入文件

# 📌 进行推理并显示进度
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
        with open("inference_results_llama.json", "w") as f:
            json.dump(results, f, indent=4)

# 确保写入剩余的结果
if len(results) % batch_size != 0:
    with open("inference_results_llama.json", "w") as f:
        json.dump(results, f, indent=4)

print("✅ 推理完成，结果已保存到 inference_results_llama.json")