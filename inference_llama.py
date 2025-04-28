import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # 进度条库

# 📌 指定 LLaMA-3.2-1B-Instruct 模型
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = "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"

# 📌 检查GPU
if not torch.cuda.is_available():
    raise RuntimeError("需要GPU支持！")
print(f"🔧 使用GPU: {torch.cuda.get_device_name(0)}")
print(f"🔧 可用GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# 📌 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    use_auth_token=HF_TOKEN,
    padding_side="left",  # 左侧填充
    truncation_side="left"  # 左侧截断
)
# 设置padding token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto",  # 自动分配到可用GPU
    use_auth_token=HF_TOKEN
)
# 确保模型的pad_token_id与tokenizer一致
model.config.pad_token_id = tokenizer.pad_token_id

# 📌 读取训练集和测试集
with open("train.json", "r") as f:
    train_data = json.load(f)

with open("test.json", "r") as f:
    test_data = json.load(f)

# 📌 创建输出目录
os.makedirs("output/llama", exist_ok=True)

# 📌 定义不同数量的few-shot示例（包括0-shot作为baseline）
few_shot_counts = [4, 6, 8, 10]

for shot_count in few_shot_counts:
    print(f"\n🔄 开始处理 {shot_count}-shot 版本")
    
    # 构建 few-shot 示例
    few_shot_examples = ""
    if shot_count > 0:  # 如果不是0-shot，则添加示例
        for i in range(shot_count):
            # few_shot_examples += f"Instruction: Analyze the following instructions and determine if they contain contradictions.\n"
            example = train_data[i]
            few_shot_examples += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        # few_shot_examples += f"Instruction: Analyze the following instructions and determine if they contain contradictions.\n"
    # else:
        # few_shot_examples = "Instruction: Analyze the following instructions and determine if they contain contradictions.\n"

    results = []
    batch_size = 8  # 或更大，取决于显存

    # 📌 进行推理并显示进度
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Processing {shot_count}-shot"):
        batch = test_data[i:i+batch_size]
        # 在每个输入前添加instruction
        batch_inputs = tokenizer([few_shot_examples + f"Input: {item['input']}\nOutput:" for item in batch], 
                               return_tensors="pt", 
                               padding=True).to(model.device)
        
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=200,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # 启用 KV 缓存
            )

        # 解码模型输出
        for idx, output in enumerate(batch_outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            
            # 提取模型的实际输出（去掉输入部分）
            input_text = few_shot_examples + f"Input: {batch[idx]['input']}\nOutput:"
            response = response.replace(input_text, "").strip()

            # 存储结果
            results.append({
                "input": batch[idx]["input"],
                "expected_output": batch[idx]["output"],  # 真实答案
                "model_output": response  # LLaMA 生成的答案
            })

        # 每5条写入一次文件
        if (i + batch_size) % batch_size == 0:
            output_file = f"output/llama/inference_results_llama_{shot_count}shot.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

    # 确保写入剩余的结果
    if len(results) % batch_size != 0:
        output_file = f"output/llama/inference_results_llama_{shot_count}shot.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"✅ {shot_count}-shot 版本完成，结果已保存到 output/llama/inference_results_llama_{shot_count}shot.json")

print("\n所有版本处理完成！")