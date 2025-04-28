import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm  # 导入 tqdm 库

# 加载基础模型和tokenizer
base_model_name = "Qwen/Qwen2.5-3B-Instruct"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    padding_side="left",
    truncation_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./lora_qwen")
model = model.merge_and_unload()  # 合并LoRA权重到基础模型

# 读取数据集
with open("train.json", "r") as f:
    train_data = json.load(f)

with open("test.json", "r") as f:
    test_data = json.load(f)

# 创建输出目录
os.makedirs("output/finetuned_qwen", exist_ok=True)

# 定义不同数量的few-shot示例
few_shot_counts = [8]

# 使用固定的验证集来构建few-shot示例
validation_examples = [
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": "If you want to go to the store, turn left at the corner. If you want to go to the store, turn right at the corner.",
        "output": "The instructions \"turn left at the corner\" and \"turn right at the corner\" are contradictory. "
    },
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": "To make a cake, first mix the flour and sugar. Then add the eggs and milk.",
        "output": "The instructions are consistent and provide a clear sequence of steps."
    },
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": "The meeting is at 2 PM. The meeting is at 3 PM.",
        "output": "The instructions \"The meeting is at 2 PM\" and \"The meeting is at 3 PM\" are contradictory. "
    },
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": "To open the door, push the handle down. To open the door, pull the handle up.",
        "output": "The instructions \"push the handle down\" and \"pull the handle up\" are contradictory. "
    },
    {
        "instruction": "Analyze the following instructions and determine if they contain contradictions.",
        "input": "To make tea, boil water and add tea leaves. Let it steep for 5 minutes.",
        "output": "The instructions are consistent and provide clear steps."
    }
]

for shot_count in few_shot_counts:
    print(f"\n🔄 开始处理 {shot_count}-shot 版本")
    
    # 构建 few-shot 示例
    few_shot_examples = ""
    if shot_count > 0:  # 如果不是0-shot，则添加示例
        for i in range(shot_count):
            example = validation_examples[i % len(validation_examples)]  # 循环使用验证集
            few_shot_examples += f"Input: {example['input']}\nOutput: {example['output']}\n\n"

    results = []
    batch_size = 8  # 增加批处理大小

    # 使用 tqdm 包装 test_data 以显示进度条
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Processing {shot_count}-shot"):
        batch = test_data[i:i+batch_size]
        
        # 批量处理输入
        batch_inputs = tokenizer(
            [few_shot_examples + f"Input: {item['input']}\nOutput:" for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # 批量生成文本
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

        # 批量解码模型输出
        for idx, output in enumerate(batch_outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            
            # 提取模型的实际输出（去掉输入部分）
            input_text = few_shot_examples + f"Input: {batch[idx]['input']}\nOutput:"
            response = response.replace(input_text, "").strip()

            # 存储结果
            results.append({
                "input": batch[idx]["input"],
                "expected_output": batch[idx]["output"],  # 真实答案
                "model_output": response  # 模型生成的答案
            })

        # 每处理完一个批次就写入文件
        output_file = f"output/finetuned_qwen/inference_results_finetuned_{shot_count}shot.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"✅ {shot_count}-shot 版本完成，结果已保存到 {output_file}")

print("\n所有版本处理完成！") 