"""生成 SCBM 二元概念向量（严格对齐 SCBM 原文）。

SCBM（Labadie-Tamayo et al., IPM 2026）使用二元概念评估，核心方法（Section 3.2）：
  1. 对每个形容词 a 和文本 t，使用简单模板构造提示词
  2. 提取 LLM 输出中"正面回应"（yes 类 token）的边际概率作为概念分数
  3. 提示词模板：Tell me if the adjective [adjective] describes the content of the following text: [text]?
  4. 适配 GLM-4-9B-Chat 中文模型：添加 System message 约束输出格式（对齐 GermEval 的 Persona 做法）

【与原始 SCBM 的主要差异】
- 原始论文使用 GPT-4o/Llama 3.1 等英文模型，此处使用 GLM-4-9B-Chat 中文模型
- 提示词翻译为中文，并添加 System message 约束输出格式（对齐 GermEval 的 Persona 做法）
- 不使用形容词定义（SCBM 原文仅在少样本 ICL 实验中引入定义）

使用示例：
  python scripts/generate_scbm_concept.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
  python scripts/generate_scbm_concept.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
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
        description="生成 SCBM 二元概念向量（vLLM 版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train: 生成训练集概念向量，test: 生成测试集概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM 模型名称')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU 显存占用比例（0.0-1.0），默认 0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表（复用 generate_concept_vectors.py 中的配置）
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
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载 vLLM 模型和 tokenizer。"""
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
# 提示词定义（对齐 SCBM 原文 Section 3.2，适配 GLM-4-9B-Chat 中文模型）
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位社会科学专家。当被问到问题时，请直接回答\"是\"或\"否\"，只回答一个词。"
)

def build_chat_messages(content, adj, adj_definition=None):
    """构建 SCBM 二元评估的 Chat Template messages。

    SCBM 原文（Section 3.2）使用简单模板：
      "Tell me if the adjective [adjective] describes the content of the following text: [text]?"
    此处翻译为中文，并适配 GLM-4-9B-Chat：
    - 保持 SCBM 原文的简单提问结构
    - 添加 System message 约束输出格式（对齐 GermEval 的 Persona 做法）
    - 不使用形容词定义（SCBM 原文仅在少样本 ICL 实验中引入定义）

    Args:
        content: 输入文本
        adj: 形容词
        adj_definition: 保留参数，SCBM 中不使用
    """
    user_content = f"告诉我，形容词\"{adj}\"是否描述了以下文本的内容：\"{content}\""

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


def extract_binary_score(first_token_logprobs, yes_ids, no_ids):
    """从首 token 的 logprobs 中提取二元概念分数。

    Args:
        first_token_logprobs: vLLM 返回的首 token logprobs 字典 {token_id: Logprob 对象}
        yes_ids: "是" 的 token id 列表
        no_ids: "否" 的 token id 列表

    Returns:
        (score, binary_probs): P("是") 归一化分数 (0~1), 二元概率列表 [P(是),P(否)]
    """
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    p_yes = sum(probs_dict.get(tid, 0.0) for tid in yes_ids)
    p_no = sum(probs_dict.get(tid, 0.0) for tid in no_ids)

    total = p_yes + p_no + 1e-8
    score = p_yes / total

    return score, [p_yes, p_no]


# =============================================================================
# 核心流程：生成 SCBM 二元概念向量
# =============================================================================
def generate_scbm_concept(data_path, output_path, csv_output_path, adjective_path,
                          tokenizer, llm_model,
                          is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成 SCBM 二元概念向量（对齐 SCBM 原文 Section 3.2）。

    对数据集中每条文本，使用 SCBM 的简单提示词模板查询每个形容词，
    通过 verbalizer 技术提取"是"类 token 的边际概率作为概念分数，
    构建概念向量（每条文本一个 V 维向量，V = 形容词数量）。

    提示词模板（对齐 SCBM 原文 Section 3.2，适配 GLM-4-9B-Chat 中文模型）：
      System: 你是一位社会科学专家。当被问到问题时，请直接回答"是"或"否"，只回答一个词。
      User: 告诉我，形容词"adj"是否描述了以下文本的内容："text"
    不使用形容词定义。
    """
    # 二元 verbalizer token
    yes_tokens = ["是"]
    no_tokens = ["否"]
    yes_ids = get_first_token_ids(yes_tokens, tokenizer)
    no_ids = get_first_token_ids(no_tokens, tokenizer)

    print(f"二元 Verbalizer token IDs:")
    print(f"  \"是\" -> {yes_ids}")
    print(f"  \"否\" -> {no_ids}")

    # 加载形容词词典（仅使用形容词本身，不使用定义 — 对齐 SCBM 原文）
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    num_adjs = len(adjectives)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # vLLM 推理
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []
    all_raw_probs = []  # 收集所有二元概率，用于验证 prompt 约束力

    for sample in tqdm(data_set, desc="Processing samples"):
        content = sample["content"]

        prompts = []
        for adj in adjectives:
            messages = build_chat_messages(content, adj)
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
            score, binary_probs = extract_binary_score(first_token_logprobs, yes_ids, no_ids)
            concept_vector.append(score)
            raw_probs.append(binary_probs)

        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector 长度异常：期望 {num_adjs}，实际 {len(concept_vector)}")

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)
        all_raw_probs.append(raw_probs)

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
    print(f"SCBM 二元概念向量 (JSON) 保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"SCBM 二元概念向量 (CSV) 保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {num_adjs}] (文本数, 形容词数)")

    # 统计概念激活率
    total_scores = len(concept_matrix) * num_adjs
    nonzero = sum(1 for row in concept_matrix for s in row if s > 0)
    coverage = nonzero / total_scores
    print(f"概念激活率: {coverage:.2%} ({nonzero}/{total_scores})")

    # 验证 prompt 约束力：P(是)+P(否) 均值
    all_p_yes = sum(p[0] for r in all_raw_probs for p in r) / total_scores
    all_p_no = sum(p[1] for r in all_raw_probs for p in r) / total_scores
    print(f"二元概率分布均值: P(是)={all_p_yes:.4f}, P(否)={all_p_no:.4f}, "
          f"P(是)+P(否)={all_p_yes+all_p_no:.4f}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    adjective_path = project_root / "data" / "raw" / "adjective" / "toxic_adjectives_v2.csv"

    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_scbm.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_scbm.csv"

    print("\n" + "=" * 60)
    print("SCBM 二元概念向量生成 (vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM 模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"JSON 输出路径: {output_path}")
    print(f"CSV 输出路径: {csv_output_path}")
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

    generate_scbm_concept(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()