"""单条文本端到端推理延迟测量脚本（LLM 概念生成 + MLP 分类）。

流程：给定一条文本 →
  1. 将该文本与 132 个概念（toxic_adjectives_v2.csv）构造 chat prompt
  2. vLLM（glm-4-9b-chat）批量生成 132 个概念的 3 级 Likert 概率
  3. 构造成 ACV-FSCBM 等级特征 x = [P(3); P(2); (P(3)-P(2))⊙w_SNR] ∈ R^{396}
  4. 加载训练好的 GatedConceptClassifier 得到预测类别

延迟口径（严格按照 LLM 已加载好的前提）：
  从"给出文本"开始计时（start = 输入文本作为原始字符串给出之后），
  直到"获取预测结果"为止的时间差作为该条文本的推理延迟。
  模型加载、SNR 计算等初始化步骤均不计时。
  重复 --repeat 次并默认取中位数作为稳健估计。

使用示例：
  python scripts/measure_inference_latency.py \
      --ckpt_dir "experiments/TOXICN(acv_fscbm)/20260912-143145_TOXICN_acv_s5114" \
      --text "你真让人恶心，滚出去。" --repeat 5

参数说明：
  --ckpt_dir                已训练 ACV-FSCBM 运行目录（含 best_model.pth 与 config.json）的相对路径，必填
  --text                    待测单条文本（不填则用内置示例文本）
  --dataset_name            数据集名称（用于定位 SNR 训练概念向量文件），默认 TOXICN
  --model_name              LLM 概念向量模型，默认 glm-4-9b-chat
  --repeat                  计时重复次数，默认 5，取中位数
  --gpu_memory_utilization  vLLM 显存占用比例，默认 0.85
  --warmup                  计时前预热的重复次数（不参与统计），默认 1
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig
from models.gated_classifier import GatedConceptClassifier

# 与 generate_concept_vectors.py 保持一致
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

DEFAULT_TEXT = "你真让人恶心，滚出去，别再出现在我面前。"


# =============================================================================
# 概念相关：词典 / 提示词 / verbalizer
# =============================================================================
def load_concepts(adjective_path):
    """加载 132 个形容词及定义。"""
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    return adjectives, definitions


def build_chat_messages(content, adj, adj_definition=None):
    """构建 3 级行为锚定 Likert 的 Chat Template messages（与生成脚本一致）。"""
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


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for w in word_list:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            token_ids.append(enc[0])
    return list(dict.fromkeys(token_ids))


def extract_3level_score(first_token_logprobs, level_ids):
    """从首 token logprobs 提取 3 级概率 [P(1), P(2), P(3)]。"""
    probs = {}
    for tid, obj in first_token_logprobs.items():
        probs[tid] = math.exp(obj.logprob)
    return [probs.get(tid, 0.0) for tid in level_ids]


# =============================================================================
# 特征构造（与 utils/train_acv_fscbm.py 的 extract_level_features 一致）
# =============================================================================
def build_level_feature(level_probs_list, snr_weights):
    """由 132 个概念的 [P1,P2,P3] 构造成 396 维等级特征。"""
    n = len(level_probs_list)
    p3 = np.array([lp[2] for lp in level_probs_list])
    p2 = np.array([lp[1] for lp in level_probs_list])
    diff = (p3 - p2) * (np.clip(snr_weights, 0, None) + 0.01)
    X = np.concatenate([p3, p2, diff])  # shape (3l,)
    return torch.tensor(X, dtype=torch.float32).unsqueeze(0)


def compute_concept_snr(train_data, n_concepts):
    """SNR 权重（与 train_acv_fscbm.compute_concept_snr 一致），初始化阶段计算，不计时。"""
    train_toxic = [i for i in train_data if i["toxic"] == 1]
    train_nt = [i for i in train_data if i["toxic"] == 0]
    snr = np.zeros(n_concepts)
    for ci in range(n_concepts):
        tc = [i["level_probs"][ci][2] - i["level_probs"][ci][1] for i in train_toxic]
        nc = [i["level_probs"][ci][2] - i["level_probs"][ci][1] for i in train_nt]
        snr[ci] = (np.mean(tc) - np.mean(nc)) / (np.std(tc + nc) + 1e-8)
    return snr


# =============================================================================
# MLP 分类器加载
# =============================================================================
def load_classifier(ckpt_dir, n_concepts, concept_types, n_main_channels):
    """从运行目录加载训练好的 GatedConceptClassifier。"""
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    use_gate = cfg.get("use_gate", True)
    model = GatedConceptClassifier(
        n_concepts, concept_types, cfg.get("dropout_rate", 0.5),
        cfg.get("hidden_features", 96), 0, n_main_channels, use_gate,
    )
    sd = torch.load(ckpt_dir / "best_model.pth", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


# =============================================================================
# 主入口
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="已训练 ACV-FSCBM 运行目录（含 best_model.pth）相对路径")
    p.add_argument("--text", default=DEFAULT_TEXT, help="待测单条文本")
    p.add_argument("--dataset_name", default="TOXICN")
    p.add_argument("--model_name", default="glm-4-9b-chat")
    p.add_argument("--repeat", type=int, default=5, help="计时重复次数，默认 5，取中位数")
    p.add_argument("--warmup", type=int, default=1, help="预热重复次数，不计入统计")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = p.parse_args()

    config = MLPConfig()
    ckpt_dir = (project_root / args.ckpt_dir).resolve()
    if not (ckpt_dir / "best_model.pth").exists():
        raise FileNotFoundError(f"未找到分类器权重: {ckpt_dir / 'best_model.pth'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 初始化阶段（不计时） ----------
    adjectives, definitions = load_concepts(config.adjective_path)
    with open(config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json", encoding="utf-8") as f:
        concept_types = [i["type"] for i in json.load(f)]
    n_concepts = len(adjectives)

    # SNR 从训练集概念向量计算（同 train_acv）
    train_file = (config.processed_path / args.dataset_name / args.model_name /
                  f"concept_train_{args.model_name}_v2_3level.json")
    with open(train_file, encoding="utf-8") as f:
        train_data = json.load(f)
    snr_weights = compute_concept_snr(train_data, n_concepts)

    # 加载 LLM（vLLM）
    print(">>> 初始化：加载 LLM...")
    llm_path = project_root / "models" / args.model_name
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(model=str(llm_path), trust_remote_code=True, dtype="auto",
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True, max_model_len=1024,
              max_num_seqs=64, max_num_batched_tokens=16384)
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    level_ids = get_first_token_ids(["1", "2", "3"], tokenizer)

    # 加载 MLP（初始化阶段，不计时）
    print(">>> 初始化：加载 MLP 分类器...")
    model, cfg = load_classifier(ckpt_dir, n_concepts, concept_types, n_main_channels=3)
    model.to(device)

    print(f">>> 文本: {args.text}")
    print(f">>> 概念数: {n_concepts}  特征维度: {n_concepts * 3}  "
          f"device: {device}  重复: {args.repeat}  预热: {args.warmup}")
    print("=" * 64)

    # 预热 + 计时（LLM 已加载；从"给出文本"开始计时）
    def infer_once(text):
        """单条文本端到端推理，返回 (延迟秒, 预测类别)。计时覆盖 prompt 构造→LLM→特征→MLP。"""
        t0 = time.perf_counter()
        prompts = []
        for adj, adj_def in zip(adjectives, definitions):
            messages = build_chat_messages(text, adj, adj_def)
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True) + "\n")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        level_probs_list = [extract_3level_score(o.outputs[0].logprobs[0], level_ids)
                            for o in outputs]
        x = build_level_feature(level_probs_list, snr_weights).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        t1 = time.perf_counter()
        return t1 - t0, pred

    # 预热
    print(f">>> 预热 {args.warmup} 次（不计入统计）...")
    for _ in range(args.warmup):
        infer_once(args.text)

    # 正式计时
    latencies, preds = [], []
    print(f">>> 测量 {args.repeat} 次...")
    for i in tqdm(range(args.repeat), desc="Timing"):
        dt, pred = infer_once(args.text)
        latencies.append(dt)
        preds.append(pred)

    med = statistics.median(latencies)
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    pred_mode = max(set(preds), key=preds.count)

    print("=" * 64)
    print(f"预测类别: {pred_mode} ({'Toxic' if pred_mode == 1 else 'Non-Toxic'})")
    for i, dt in enumerate(latencies, 1):
        print(f"  第 {i} 次: {dt * 1000:.1f} ms")
    print(f"{'=' * 64}")
    print(f"单条文本端到端推理延迟（LLM 已加载）")
    print(f"  中位数: {med * 1000:.1f} ms")
    print(f"  均值:   {mean * 1000:.1f} ms  ± {std * 1000:.1f} ms")
    print(f"  （N={args.repeat}，含 132 个概念一次批量生成 + 特征 + MLP 前向）")


if __name__ == "__main__":
    main()