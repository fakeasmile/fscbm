"""使用 vLLM 生成 3 级行为锚定 Likert 概念向量。

对应论文 3.2 节概念向量生成：
  3 级 Likert 量表（1=不相关, 2=涉及, 3=实施），
  通过 verbalizer 技术提取 P(3) 归一化分数作为概念激活度，
  同时保存完整 3 级概率分布以支持双维度表示。

使用示例：
  python scripts/generate_concept_vectors.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
  python scripts/generate_concept_vectors.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


# =============================================================================
# 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 3 级行为锚定 Likert 概念向量（vLLM 版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train: 生成训练集概念向量，test: 生成测试集概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM 模型名称')
    parser.add_argument('--adjective_name', type=str, default=None,
                        help='形容词词典文件名（默认使用 MLP_config.py 中的 adjective_path）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU 显存占用比例（0.0-1.0），默认 0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen2.5-14B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",
        "is_qwen3": True,
        "is_multimodal": True,
        "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "\n",
    },
    "deepseek-llm-7b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Baichuan2-7B-Chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3-8B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
}


def get_model_loading_config(model_name: str) -> dict:
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置，未知模型直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载 vLLM 模型和 tokenizer。

    模型差异（量化、多模态、Qwen3+）均从 MODEL_LOADING_CONFIG 读取。
    Returns: (tokenizer, llm, is_qwen3)
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已设置 limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的言语行为意图。\n"
    "你的任务是判断文本中说话者与指定概念之间的关系，从三个级别中选择最匹配的：\n"
    "1 — 不相关：文本与该概念描述的内容无关\n"
    "2 — 涉及：文本涉及该概念，但说话者是在提及、讨论、分析、引用或反驳，而非自身实施\n"
    "3 — 实施：说话者本人正在实施该概念描述的言行\n"
    "\"涉及\"与\"实施\"的关键区分：\n"
    "- 涉及：说话者在谈论该行为，但自身并非行为的执行者\n"
    "  例如：分析辱骂现象、报告歧视事件、引用他人攻击性言论、反驳偏见观点\n"
    "- 实施：说话者本人就是该行为的执行者\n"
    "  例如：说话者本人正在辱骂、威胁、歧视、煽动，或以比喻、反讽、谐音等隐含方式实施\n"
    "请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等）。\n"
    "只回答一个数字（1、2或3）。"
)


# =============================================================================
# Prompt 构建
# =============================================================================
def build_chat_messages(content, adj, adj_definition=None):
    """构建 3 级行为锚定 Likert 的 Chat Template messages。

    user_content 结构：
        文本内容：{content}
        概念：{adj}
        定义：{adj_definition}  ← 仅当定义存在时插入
        说话者与"{adj}"的关系级别（1=不相关 2=涉及 3=实施）：回答：
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"概念：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"说话者与\"{adj}\"的关系级别（1=不相关 2=涉及 3=实施）：回答：")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer 工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首 token id（去重）。"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效 token")
    return list(dict.fromkeys(token_ids))


def extract_3level_score(first_token_logprobs, level_ids):
    """从首 token 的 logprobs 中提取 3 级 Likert 分数。

    Args:
        first_token_logprobs: vLLM 返回的首 token logprobs 字典 {token_id: Logprob 对象}
        level_ids: 3 级 verbalizer token id 列表（对应 "1","2","3"）

    Returns:
        (score, level_probs): P(3) 归一化分数 (0~1), 3 级概率列表 [P(1),P(2),P(3)]
    """
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in level_ids]

    total = sum(level_probs) + 1e-8
    score = level_probs[2] / total  # P(3) 归一化

    return score, level_probs


# =============================================================================
# 核心流程：生成 3 级行为锚定概念向量
# =============================================================================
def generate_3level_concept(data_path, output_path, csv_output_path, adjective_path,
                            tokenizer, llm_model,
                            is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成 3 级行为锚定 Likert 概念向量。

    对数据集中每条文本，遍历所有形容词，通过 verbalizer 技术提取 3 级 Likert 评分，
    构建概念向量（每条文本一个 V 维向量，V = 形容词数量）。

    流程：
    1. 加载形容词词典和数据集
    2. 逐文本处理：构建 prompt → vLLM 推理 → 提取首 token 概率 → P(3) 归一化分数
    3. 保存结果（JSON 含完整信息，CSV 为纯矩阵）
    """
    # 3 级 Likert verbalizer token id
    level_tokens = ["1", "2", "3"]
    level_ids = get_first_token_ids(level_tokens, tokenizer)

    print(f"3 级 Verbalizer token IDs: {level_ids}")
    for token, tid in zip(level_tokens, level_ids):
        print(f"  \"{token}\" -> {tid}")

    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # vLLM 推理
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample in tqdm(data_set, desc="Processing samples"):
        content = sample["content"]

        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(content, adj, adj_def)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_3level_score(first_token_logprobs, level_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector 长度异常：期望 {num_adjs}，实际 {len(concept_vector)}")

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "level_probs": raw_probs,
        }
        results.append(result_item)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"3 级行为锚定概念向量 (JSON) 保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"3 级行为锚定概念向量 (CSV) 保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {num_adjs}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为 0")

    # 统计 verbalizer 覆盖率和 3 级分布
    total_scores = len(concept_matrix) * num_adjs
    nonzero = sum(1 for row in concept_matrix for s in row if s > 0)
    coverage = nonzero / total_scores

    all_p1 = sum(p[0] for r in results for p in r["level_probs"]) / total_scores
    all_p2 = sum(p[1] for r in results for p in r["level_probs"]) / total_scores
    all_p3 = sum(p[2] for r in results for p in r["level_probs"]) / total_scores

    print(f"概念激活率: {coverage:.2%} ({nonzero}/{total_scores})")
    print(f"3 级概率分布均值: P(1)={all_p1:.4f}, P(2)={all_p2:.4f}, P(3)={all_p3:.4f}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    if args.adjective_name is not None:
        adjective_path = config.raw_data_path / "adjective" / args.adjective_name
    else:
        adjective_path = config.adjective_path
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    adj_stem = adjective_path.stem
    adj_version = adj_stem.replace("toxic_adjectives_", "")

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_3level.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_3level.csv"

    print("\n" + "=" * 60)
    print("3 级行为锚定 Likert 概念向量生成 (vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM 模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name} ({adjective_path})")
    print(f"当前模式: {args.mode}")
    print(f"GPU 显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"JSON 输出路径: {output_path}")
    print(f"CSV 输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到 Qwen3+ 模型 ({args.model_name})，已禁用思考模式 (enable_thinking=False)")

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型 ({args.model_name}) 需要追加 prompt 后缀: {repr(prompt_suffix)}")

    generate_3level_concept(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()