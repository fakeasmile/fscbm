"""SCBM verbalizer 覆盖率全景分析工具（全形容词扫描，vLLM 版本）

【定位】
本脚本是 generate_scbm_concept.py 的"全形容词切片"评估工具。
对一条固定文本遍历所有 132 个形容词，评估 SCBM 提示词模板和二元 verbalizer
在全部形容词词典上的覆盖能力是否稳定。

【核心指标】
- total_prob = P("是") + P("否")：衡量 prompt 对 LLM 输出的约束力
  理想情况下应接近 1.0，说明 LLM 首 token 一定是"是"或"否"
- first_token_text：LLM 实际输出的首 token，用于验证 prompt 是否将回答约束在 verbalizer 词表内

【输出】
1. 可视化图表（PNG）：横轴为形容词索引，纵轴为概率值
   - total_prob 蓝线 + "是"概率 绿线 + "否"概率 橙线
2. JSON 数据文件：每个形容词的详细概率数据 + 统计摘要

【使用方法】
1. 修改下方 CONFIG 区域的变量
2. 运行：python scripts/inspect_scbm_verbalizer.py
"""
import json
import math
import os
import sys
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "glm-4-9b-chat"  # models目录下的模型文件夹名

# 文本内容（直接修改即可）
TEXT_CONTENT = "什么被害妄想猎巫man"

# 输出目录（相对于项目根目录）
OUTPUT_DIR = "experiments/verbalizer_coverage"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
# ===================================================================


# 模型加载配置表（与 generate_scbm_concept.py 保持一致）
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
    """加载 vLLM 模型和 tokenizer（复用 generate_scbm_concept 逻辑）"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True, padding_side="right",
    )
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


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首 token id（去重）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效 token")
    return list(dict.fromkeys(token_ids))


# =============================================================================
# 提示词定义（对齐 SCBM 原文 Section 3.2，与 generate_scbm_concept.py 一致）
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位社会科学专家。当被问到问题时，请直接回答\"是\"或\"否\"，只回答一个词。"
)

def build_chat_messages(content, adj):
    """构建 SCBM 二元评估的 Chat Template messages。

    与 generate_scbm_concept.py 中的 build_chat_messages 完全一致。
    """
    user_content = f"告诉我，形容词\"{adj}\"是否描述了以下文本的内容：\"{content}\""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def analyze_scbm_verbalizer_coverage(
    text_content,
    adjective_path,
    tokenizer,
    llm_model,
    output_dir: Path,
    model_name: str,
    is_qwen3=False,
    prompt_suffix="",
):
    """对单条文本遍历所有形容词，评估 SCBM 二元 verbalizer 覆盖率。"""
    # 二元 verbalizer token
    yes_tokens = ["是"]
    no_tokens = ["否"]
    yes_ids = get_first_token_ids(yes_tokens, tokenizer)
    no_ids = get_first_token_ids(no_tokens, tokenizer)
    all_verbalizer_ids = set(yes_ids + no_ids)

    print(f"二元 Verbalizer token IDs:")
    print(f"  \"是\" -> {yes_ids}")
    print(f"  \"否\" -> {no_ids}")

    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_en_list = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adjectives)

    # vLLM 采样配置
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 构建所有提示词
    prompts = []
    for adj in adjectives:
        messages = build_chat_messages(text_content, adj)
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # 批量推理
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    # 存储结果
    first_tokens = []  # 记录所有首 token 文本
    results = []

    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0]

        # 提取首 token 信息
        first_token_id = sample_info.outputs[0].token_ids[0]
        first_token_text = tokenizer.decode([first_token_id]).strip()
        first_tokens.append(first_token_text)

        # 转换为概率字典
        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        # 提取 verbalizer 概率
        p_yes = sum(probs_dict.get(tid, 0.0) for tid in yes_ids)
        p_no = sum(probs_dict.get(tid, 0.0) for tid in no_ids)
        total_prob = p_yes + p_no

        # 首 token 是否在 verbalizer 词表中
        in_verbalizer = first_token_id in all_verbalizer_ids

        # verbalizer 归一化分数
        score = p_yes / (total_prob + 1e-8)

        results.append({
            "index": adj_idx,
            "adjective_en": adj_en_list[adj_idx],
            "adjective_cn": adjectives[adj_idx],
            "p_yes": round(p_yes, 6),
            "p_no": round(p_no, 6),
            "total_prob": round(total_prob, 6),
            "score": round(score, 6),
            "first_token_id": first_token_id,
            "first_token_text": first_token_text,
            "in_verbalizer": in_verbalizer,
        })

    # 首 token 分布统计
    token_dist = Counter(first_tokens)
    verbalizer_coverage = sum(1 for t in first_tokens if t in ("是", "否"))
    verbalizer_pct = verbalizer_coverage / len(first_tokens) * 100

    # 保存 JSON 数据
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_text = text_content[:20].replace("\\", "").replace("/", "").replace(" ", "_")
    json_path = output_dir / f"scbm_{safe_text}_{model_name}_vllm.json"
    stats = {
        "model_name": model_name,
        "template": "scbm_binary",
        "text_content": text_content,
        "num_adjectives": len(adjectives),
        "verbalizer_coverage_pct": round(verbalizer_pct, 2),
        "first_token_distribution": {k: v for k, v in sorted(token_dist.items(), key=lambda x: -x[1])},
        "statistics": {
            "mean_total_prob": round(sum(r["total_prob"] for r in results) / len(results), 6),
            "min_total_prob": round(min(r["total_prob"] for r in results), 6),
            "max_total_prob": round(max(r["total_prob"] for r in results), 6),
            "mean_p_yes": round(sum(r["p_yes"] for r in results) / len(results), 6),
            "mean_p_no": round(sum(r["p_no"] for r in results) / len(results), 6),
        },
        "data": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"数据已保存: {json_path}")

    # 绘制图表
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 左图：verbalizer 概率曲线
    ax1 = axes[0]
    x = [r["index"] for r in results]
    total_probs = [r["total_prob"] for r in results]
    p_yes_list = [r["p_yes"] for r in results]
    p_no_list = [r["p_no"] for r in results]
    ax1.plot(x, total_probs, label="total_prob (P(是)+P(否))", color="blue", alpha=0.9, linewidth=1.2)
    ax1.plot(x, p_yes_list, label="P(是)", color="green", alpha=0.7, linewidth=0.8, linestyle="--")
    ax1.plot(x, p_no_list, label="P(否)", color="orange", alpha=0.7, linewidth=0.8, linestyle="--")
    mean_total = sum(total_probs) / len(total_probs)
    ax1.axhline(y=mean_total, color="blue", linestyle=":", alpha=0.5, label=f"total均值: {mean_total:.3f}")
    ax1.set_xlabel("形容词索引", fontsize=12)
    ax1.set_ylabel("概率", fontsize=12)
    ax1.set_title(
        f"SCBM Verbalizer 覆盖率分析\n模型: {model_name} | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, len(adjectives) - 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    # 底部形容词标签（稀疏显示）
    tick_step = max(1, len(adjectives) // 20)
    tick_positions = list(range(0, len(adjectives), tick_step))
    tick_labels = [adjectives[i] if i < len(adjectives) else "" for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    # 右图：首 token 分布（饼图）
    ax2 = axes[1]
    colors = {"是": "#4CAF50", "否": "#FF9800"}
    other_tokens = {k: v for k, v in token_dist.items() if k not in ("是", "否")}
    if other_tokens:
        labels = ["是", "否", "其他"]
        sizes = [token_dist.get("是", 0), token_dist.get("否", 0), sum(other_tokens.values())]
        pie_colors = [colors.get("是", "#4CAF50"), colors.get("否", "#FF9800"), "#E0E0E0"]
        explode = (0, 0, 0.05)
        label_detail = f"其他: {dict(sorted(other_tokens.items(), key=lambda x: -x[1])[:5])}"
    else:
        labels = ["是", "否"]
        sizes = [token_dist.get("是", 0), token_dist.get("否", 0)]
        pie_colors = [colors.get("是", "#4CAF50"), colors.get("否", "#FF9800")]
        explode = (0, 0)
        label_detail = ""

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=pie_colors, explode=explode,
        startangle=90, textprops={"fontsize": 12},
    )
    ax2.set_title(
        f"首 Token 分布\nVerbalizer 覆盖率: {verbalizer_pct:.1f}%",
        fontsize=14,
    )
    if label_detail:
        ax2.text(0.5, -0.15, label_detail, transform=ax2.transAxes,
                 ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    png_path = output_dir / f"scbm_{safe_text}_{model_name}_vllm.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {png_path}")
    plt.close()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("SCBM Verbalizer 覆盖率统计摘要")
    print("=" * 60)
    print(f"形容词数量: {len(adjectives)}")
    print(f"Verbalizer 覆盖率: {verbalizer_pct:.2f}%")
    print(f"  首 token 为\"是\": {token_dist.get('是', 0)} ({token_dist.get('是', 0)/len(adjectives)*100:.1f}%)")
    print(f"  首 token 为\"否\": {token_dist.get('否', 0)} ({token_dist.get('否', 0)/len(adjectives)*100:.1f}%)")
    if other_tokens:
        print(f"  其他 token: {dict(sorted(other_tokens.items(), key=lambda x: -x[1])[:10])}")
    print(f"total_prob 均值: {stats['statistics']['mean_total_prob']:.4f}")
    print(f"total_prob 最小值: {stats['statistics']['min_total_prob']:.4f}")
    print(f"total_prob 最大值: {stats['statistics']['max_total_prob']:.4f}")
    print(f"P(是) 均值: {stats['statistics']['mean_p_yes']:.4f}")
    print(f"P(否) 均值: {stats['statistics']['mean_p_no']:.4f}")
    print("=" * 60)

    return results


def main():
    config = MLPConfig()
    output_dir = project_root / OUTPUT_DIR

    print("\n" + "=" * 60)
    print("SCBM Verbalizer 覆盖率分析（vLLM 版本）")
    print("=" * 60)
    print(f"模型名称: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        project_root / "models", MODEL_NAME, GPU_MEMORY_UTILIZATION
    )
    if qwen3_flag:
        print(f"检测到 Qwen3+ 模型({MODEL_NAME})，已禁用思考模式")
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({MODEL_NAME})需要追加 prompt 后缀: {repr(prompt_suffix)}")

    analyze_scbm_verbalizer_coverage(
        text_content=TEXT_CONTENT,
        adjective_path=project_root / "data" / "raw" / "adjective" / "toxic_adjectives_v2.csv",
        tokenizer=tokenizer,
        llm_model=llm_model,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix,
    )


if __name__ == "__main__":
    main()