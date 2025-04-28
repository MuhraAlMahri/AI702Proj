import os
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from tqdm import tqdm

# 关键词列表
keywords = [
    "contradictory", "contradiction", "conflict", "inconsistent", "inconsistency", "paradox",
    "self-contradictory", "paradoxical", "mutually exclusive", "opposing", "discrepancy", 
    "incongruous", "disagreement", "logical fallacy", "circular reasoning", "doublethink",
    "oxymoron", "ambiguous", "contravening", "discordant", "irreconcilable", "duality",
    "counterintuitive", "contradistinction", "antithetical", "incoherent", "dissonance"
]

def evaluate_file(file_path):
    """评估单个文件的结果"""
    print(f"\n📊 评估文件: {file_path}")
    
    # 读取结果文件
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            results = json.load(f)
    elif file_path.endswith('.csv'):
        results = pd.read_csv(file_path).to_dict('records')
    else:
        print(f"❌ 不支持的文件格式: {file_path}")
        return None

    # 评估指标
    bleu_scores = []
    keyword_counts = Counter()

    for item in tqdm(results, desc="处理中"):
        ref = item["expected_output"].lower()  # 真实答案
        pred = item["model_output"].lower()  # 模型生成的答案

        # 计算 BLEU 分数
        bleu = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(bleu)

        # 计算关键词匹配
        found = any(keyword in pred for keyword in keywords)
        if 'instruction: analyze the following text in input and determine if they contain contradictions,' in pred:
            found = False
            print('found', found)
        if found:
            keyword_counts["recognized"] += 1
        else:
            keyword_counts["missed"] += 1

    # 计算最终分数
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    total = keyword_counts["recognized"] + keyword_counts["missed"]
    recognition_rate = (keyword_counts["recognized"] / total * 100) if total > 0 else 0.0

    return {
        "file": os.path.basename(file_path),
        "avg_bleu": avg_bleu,
        "recognized": keyword_counts["recognized"],
        "missed": keyword_counts["missed"],
        "total": total,
        "recognition_rate": recognition_rate
    }

def main():
    # 设置输出文件夹路径
    output_dir = "output/finetuned_qwen"
    
    # 获取所有结果文件
    result_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.json', '.csv')):
                result_files.append(os.path.join(root, file))

    if not result_files:
        print(f"❌ 在 {output_dir} 目录下没有找到结果文件")
        return

    print(f"📂 找到 {len(result_files)} 个结果文件")

    # 评估所有文件
    results = []
    for file_path in result_files:
        result = evaluate_file(file_path)
        if result:
            results.append(result)

    # 打印汇总结果
    print("\n📈 Evaluation Results Summary:")
    print("-" * 80)
    print(f"{'Filename':<30} {'BLEU Score':<10} {'Recognition Rate':<10} {'Recognized/Total':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['file']:<30} {result['avg_bleu']:.4f}    {result['recognition_rate']:.2f}%    {result['recognized']}/{result['total']}")

if __name__ == "__main__":
    main() 