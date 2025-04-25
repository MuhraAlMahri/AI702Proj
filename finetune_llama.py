from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import datasets

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = "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"

# 1️⃣ 加载 LLaMA 3.2-1B-Instruct
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", use_auth_token=HF_TOKEN)

# 2️⃣ 加载数据集（train.json）
dataset = datasets.load_dataset("json", data_files="train.json")["train"]
print(len(dataset))

def format_function(examples):
    prompts = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {outp}"
        for instr, inp, outp in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    # 编码输入
    model_inputs = tokenizer(prompts, truncation=True, max_length=512, padding="max_length")
    # 设置标签，复制 input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# 批量 Tokenization
dataset = dataset.map(format_function, batched=True, remove_columns=dataset.column_names)

# 3️⃣ 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言建模
    r=16,  # LoRA rank，控制参数减少的程度（16 是推荐值）
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05  # 防止过拟合
)

# 4️⃣ 让模型使用 LoRA
model = get_peft_model(model, lora_config)

# 5️⃣ 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # 减小批量大小以避免内存问题
    gradient_accumulation_steps=8,   # 增加梯度累积步数
    warmup_steps=100,               # 添加预热步骤
    learning_rate=2e-4,            # 设置学习率
    fp16=True,                     # 使用混合精度训练
    output_dir="./llama3-finetune",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    num_train_epochs=5,
    report_to="none",
    label_names=["labels"]
)

# 6️⃣ 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset
)
trainer.train()

# 7️⃣ 保存微调模型
model.save_pretrained("./lora_llama3")
tokenizer.save_pretrained("./lora_llama3")

print("Fine-tuning complete! Model saved at './lora_llama3'")