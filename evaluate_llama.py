import json
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter

# 📌 关键词列表
keywords = [
    "contradictory", "contradiction", "conflict", "inconsistent", "inconsistency", "paradox",
    "self-contradictory", "paradoxical", "mutually exclusive", "opposing", "discrepancy", 
    "incongruous", "disagreement", "logical fallacy", "circular reasoning", "doublethink",
    "oxymoron", "ambiguous", "contravening", "discordant", "irreconcilable", "duality",
    "counterintuitive", "contradistinction", "antithetical", "incoherent", "dissonance"
]

# 📌 读取推理结果
with open("inference_results_llama.json", "r") as f:
    results = json.load(f)

# 📌 评估指标
bleu_scores = []
keyword_counts = Counter()

for item in results:
    ref = item["expected_output"].lower()  # 真实答案 (转小写，防止大小写影响)
    pred = item["model_output"].lower()  # LLaMA 生成的答案 (转小写)

    # 1️⃣ 计算 BLEU 分数
    bleu = sentence_bleu([ref.split()], pred.split())  # 以单词为单位计算 BLEU
    bleu_scores.append(bleu)

    # 2️⃣ 计算关键词匹配
    found = any(keyword in pred for keyword in keywords)
    if found:
        keyword_counts["recognized"] += 1
    else:
        keyword_counts["missed"] += 1

# 📌 计算最终分数
avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
total = keyword_counts["recognized"] + keyword_counts["missed"]
recognition_rate = (keyword_counts["recognized"] / total * 100) if total > 0 else 0.0

# 📌 打印结果
print(f"✅ 评估完成！")
print(f"🔹 平均 BLEU 分数: {avg_bleu:.4f}")
print(f"🔹 关键词匹配成功的样本: {keyword_counts['recognized']} / {total} ({recognition_rate:.2f}%)")