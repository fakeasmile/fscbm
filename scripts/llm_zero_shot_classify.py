"""LLM Zero-Shot 分类基线（参考 SCBM 论文 Table 4 Category IV）。

直接使用 LLM 进行有害言论二分类，不经过形容词概念瓶颈。
通过 verbalizer 技术提取"是"/"否"首 token 概率作为分类依据。

【对比目的】
SCBM 论文 Table 4 中将 GPT-3.5/GPT-4o 的 zero-shot 分类作为 Category IV 基线，
与 SCBM（Category V）对比，证明概念瓶颈方法的优势。
本脚本复现这一对比：LLM 直接分类 vs LLM 概念评分 + 下游分类器。

【prompt 设计原则】
- 角色定义与概念评分提示词一致（"语言分析专家"）
- 包含有害言论的操作化定义（参考顶刊prompt设计）
- 要求考虑隐含语义（比喻、反讽、谐音等）
- 严格限定输出格式（只回答"是"或"否"）

使用示例：
  python scripts/llm_zero_shot_classify.py --dataset_name TOXICN --model_name glm-4-9b-chat
  python scripts/llm_zero_shot_classify.py --dataset_name COLD --model_name glm-4-9b-chat --mode test
"""

import argparse
import math
import os
import sys
from pathlib import Path
from datetime import datetime
import json

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import precision_recall_fscore_support, classification_report

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig
from scripts.generate_concept_vectors import (
    MODEL_LOADING_CONFIG,
    get_model_loading_config,
    load_vllm_model,
    get_first_token_ids,
)


# =============================================================================
# Prompt 定义
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "你的任务是判断文本是否为有害言论。\n"
    "有害言论指：针对特定群体或个人，基于其身份特征（如种族、民族、宗教、性别、性取向等）\n"
    "进行攻击、侮辱、贬低、威胁或煽动敌意的言论。\n"
    "请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等）。\n"
    "直接回答\"是\"或\"否\"，只回答一个词。"
)


def build_zero_shot_messages(content):
    """构建 Zero-Shot 分类 Chat Template。

    user_content 使用与概念评分一致的结构：
        文本内容：{content}
        该文本是否为有害言论？回答：
    """
    user_content = f"文本内容：{content}\n该文本是否为有害言论？回答："
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# 核心流程：Zero-Shot 分类
# =============================================================================
def zero_shot_classify(data_path, output_dir, tokenizer, llm_model,
                       is_qwen3=False, prompt_suffix="", model_name=""):
    """使用 LLM 进行 zero-shot 二分类。

    流程：
    1. 加载数据集
    2. 逐文本构建 prompt → vLLM 推理（max_tokens=1, temperature=0）
    3. 提取首 token 的 verbalizer 概率（"是" vs "否"）
    4. 计算分类指标并保存

    Args:
        data_path: 数据集路径（JSON）
        output_dir: 输出目录
        tokenizer: HuggingFace tokenizer
        llm_model: vLLM 模型
        is_qwen3: 是否为 Qwen3+ 模型（需禁用思考模式）
        prompt_suffix: 模型特定的 prompt 后缀
        model_name: 模型名称（用于日志）
    """
    # Verbalizer：仅使用中文"是"/"否"（目标数据集为中文）
    yes_tokens = ["是"]
    no_tokens = ["否"]
    yes_ids = get_first_token_ids(yes_tokens, tokenizer)
    no_ids = get_first_token_ids(no_tokens, tokenizer)

    print(f"Verbalizer token IDs:")
    print(f"  \"是\" -> {yes_ids}")
    print(f"  \"否\" -> {no_ids}")

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)
    print(f"数据集大小: {len(data_set)}")

    # vLLM 采样配置：max_tokens=1 强制单 token 输出
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 构建所有 prompt
    prompts = []
    for sample in data_set:
        messages = build_zero_shot_messages(sample["content"])
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # 批量推理
    print("开始批量推理...")
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=True)

    # 提取结果
    results = []
    pred_labels = []
    true_labels = []

    for idx, (sample, output) in enumerate(zip(data_set, outputs)):
        # 首 token 的 logprobs
        first_token_logprobs = output.outputs[0].logprobs[0]

        # 将 logprobs 转为概率字典
        probs_dict = {}
        for tid, logprob_obj in first_token_logprobs.items():
            probs_dict[tid] = math.exp(logprob_obj.logprob)

        # 提取"是"和"否"的概率
        p_yes = sum(probs_dict.get(tid, 0.0) for tid in yes_ids)
        p_no = sum(probs_dict.get(tid, 0.0) for tid in no_ids)

        # 归一化
        total = p_yes + p_no + 1e-8
        p_yes_norm = p_yes / total
        p_no_norm = p_no / total

        # 预测：P("是") > 0.5 判定为有毒
        pred_label = 1 if p_yes_norm > 0.5 else 0
        true_label = sample["toxic"]

        pred_labels.append(pred_label)
        true_labels.append(true_label)

        # 实际生成的 token（用于验证 prompt 约束力）
        actual_token_id = output.outputs[0].token_ids[0]
        actual_token_text = tokenizer.decode([actual_token_id])

        # 首 token 位置的 top-5 候选（用于检查 verbalizer 覆盖率）
        top5 = sorted(
            first_token_logprobs.items(),
            key=lambda x: x[1].logprob, reverse=True
        )[:5]
        top5_tokens = [
            {"token_id": tid, "token": tokenizer.decode([tid]),
             "logprob": round(obj.logprob, 4), "prob": round(math.exp(obj.logprob), 6)}
            for tid, obj in top5
        ]

        results.append({
            "index": idx,
            "content": sample["content"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": pred_label == true_label,
            "p_yes": round(p_yes_norm, 6),
            "p_no": round(p_no_norm, 6),
            "first_token_id": actual_token_id,
            "first_token_text": actual_token_text,
            "top5_candidates": top5_tokens,
        })

    # 计算指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )

    # 打印结果
    report_str = classification_report(true_labels, pred_labels, target_names=["Non-Toxic", "Toxic"])
    print("\n" + "=" * 60)
    print(f"LLM Zero-Shot 分类结果 ({model_name})")
    print("=" * 60)
    print(f"Macro F1:     {f1_macro:.4f}")
    print(f"Precision:    {precision_macro:.4f}")
    print(f"Recall:       {recall_macro:.4f}")
    print(f"Weighted F1:  {f1_weighted:.4f}")
    print("------------------------------")
    print("详细分类报告:")
    print(report_str)
    print("=" * 60)

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name, "dataset": output_dir.parent.name,
        "mode": "zero_shot", "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "f1_macro": float(f1_macro), "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_weighted": float(f1_weighted),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"LLM Zero-Shot 分类结果 ({model_name})\n")
        f.write("=" * 30 + "\n")
        f.write(f"F1 (Macro): {f1_macro:.4f}\n")
        f.write(f"Precision:  {precision_macro:.4f}\n")
        f.write(f"Recall:     {recall_macro:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(report_str + "\n")

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n结果已保存到: {output_dir}")
    print(f"  metrics.json")
    print(f"  classification_report.txt")
    print(f"  predictions.json")


# =============================================================================
# 命令行入口
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Zero-Shot 分类基线（参考 SCBM Table 4 Category IV）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["TOXICN", "COLD"], help="数据集名称")
    parser.add_argument("--model_name", type=str, required=True,
                        help="LLM 模型名称")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["train", "test"], help="数据集划分")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU 显存占用比例（0.0-1.0）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    output_dir = config.experiment_path / f"{args.dataset_name}_{args.model_name}_zero_shot_{args.mode}"

    print("\n" + "=" * 60)
    print("LLM Zero-Shot 分类基线 - 配置信息")
    print("=" * 60)
    print(f"数据集: {args.dataset_name} ({args.mode})")
    print(f"模型:   {args.model_name}")
    print(f"数据:   {data_path}")
    print(f"输出:   {output_dir}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到 Qwen3+ 模型，已禁用思考模式")

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到 prompt 后缀: {repr(prompt_suffix)}")

    zero_shot_classify(
        data_path, output_dir, tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, model_name=args.model_name,
    )

    print("Zero-Shot 分类完成。")