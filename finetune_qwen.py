from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# 1️⃣ 加载 Qwen2.5-3B-Instruct
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型并准备量化训练
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    },
    use_cache=False,  # 明确禁用缓存
    attn_implementation="flash_attention_2"  # 使用新的配置方式
)
model = prepare_model_for_kbit_training(model)

# 2️⃣ 加载数据集（train.json）
dataset = datasets.load_dataset("json", data_files="train.json")["train"]
print(f"数据集大小: {len(dataset)}")

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
dataset = dataset.map(
    format_function, 
    batched=True, 
    batch_size=64,  # 增加批处理大小
    remove_columns=dataset.column_names,
    num_proc=8  # 增加进程数
)

# 3️⃣ 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言建模
    r=16,  # LoRA rank，控制参数减少的程度（16 是推荐值）
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05,  # 防止过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 使用标准的注意力模块名称
    bias="none",  # 不微调偏置
    modules_to_save=None  # 不保存其他模块
)

# 4️⃣ 让模型使用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# 确保模型处于训练模式
model.train()

# 5️⃣ 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # 增加批量大小
    gradient_accumulation_steps=4,   # 减少梯度累积步数
    warmup_steps=100,               # 添加预热步骤
    learning_rate=2e-4,            # 设置学习率
    bf16=True,                     # 使用 bfloat16 精度
    output_dir="./qwen-finetune",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    num_train_epochs=5,
    report_to="none",
    label_names=["labels"],
    dataloader_num_workers=8,      # 增加工作进程数
    dataloader_pin_memory=True,    # 使用固定内存
    gradient_checkpointing=True,   # 使用梯度检查点以节省显存
    optim="adamw_torch_fused",    # 使用融合的 AdamW 优化器
    lr_scheduler_type="cosine",   # 使用余弦学习率调度器
    weight_decay=0.01,            # 添加权重衰减
    max_grad_norm=1.0,            # 梯度裁剪
    torch_compile=True,           # 使用 torch.compile 加速
    torch_compile_mode="max-autotune"  # 使用最大自动调优
)

# 6️⃣ 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,  # 添加 tokenizer 以启用自动填充
)

# 使用 tqdm 显示训练进度
trainer.train()

# 7️⃣ 保存微调模型
model.save_pretrained("./lora_qwen")
tokenizer.save_pretrained("./lora_qwen")

print("Fine-tuning complete! Model saved at './lora_qwen'") 