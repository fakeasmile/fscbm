# -*- coding: utf-8 -*-
"""概念决策忠实度评估（Comprehensiveness & Sufficiency）—— 对应论文 2.3.4 节「决策忠实度评估」，产出表7。

【CBM 标准做法】采用 CBM 决策忠实度（decision faithfulness）的标准做法——ERASER
（DeYoung et al., 2020, ACL）的完备性/充分性（comprehensiveness / sufficiency）指标，
将其迁移到概念层：把概念视为概念瓶颈层的「软 rationale」，量化「概念排序是否忠实反映模型决策」。
全程不依赖人工标注，仅利用模型对门控后概念表示的因果依赖，契合「免概念标注 CBM」范式。

概念瓶颈：x = [p^{(3)}; p^{(2)}; d̃] ∈ R^{3L}，门控后表示 x̃ = x ⊙ σ(gate(x)) 即
下游分类器实际使用的瓶颈层。概念 c_i 的 3 个通道位于列 {c_i, c_i+L, c_i+2L}。

指标（以有毒类概率 p_toxic = softmax(logits)[:,1] 为测量对象）：
  1. 完备性 Comprehensiveness：移除按"梯度×输入归因"排序的 top 概念后 p_toxic 的下降，
     越大说明被移除概念越关键。
  2. 充分性 Sufficiency：仅保留 top 概念后 p_toxic 与完整输入的差距，越小越充分。
  3. 显著性检验：配对置换检验（逐样本的 top 排序 vs 随机排序效果差），
     单侧 p 值，验证 top 排序显著优于随机排序。

【需要的数据/目录】（均在项目根目录 D:/fscbm 下，须已离线生成完毕，运行时不调用 LLM）：
  1. LLM 三级评分概率：
     D:/fscbm/data/processed/TOXICN/glm-4-9b-chat/concept_train_glm-4-9b-chat_v2_3level.json
     D:/fscbm/data/processed/TOXICN/glm-4-9b-chat/concept_test_glm-4-9b-chat_v2_3level.json
  2. 形容词类型定义（132 个形容词的类型标记）：
     D:/fscbm/data/raw/adjective/toxic_adjectives_v2_types.json
  3. 已训练门控分类器权重（5 个 seed 集成，逐 seed 读 best_model.pth）：
     D:/fscbm/experiments/TOXICN(acv_fscbm)/<时间戳>_TOXICN_acv_s{seed}/best_model.pth
     （seed ∈ {4468, 729, 1196, 5114, 1136}，由 resolve_model_dir 自动匹配“时间戳”目录）
  4. 毒性标签（0/1）随概念 json 一并读取（te_y），无需额外文件。

【不需要的数据】不读取任何「文本-概念等级」的人工标注，尤其不读
annotation/concept_fidelity_manual.csv。

【用法】在 D:/fscbm 下，toxicnew 环境：
  python xai/concept_faithfulness.py --exp-dir "experiments/TOXICN(acv_fscbm)" --dataset TOXICN

【输出】控制台 + xai/result/concept_faithfulness.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path("d:/fscbm")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.MLP_config import MLPConfig
from models.gated_classifier import GatedConceptClassifier
from utils.train_acv_fscbm import compute_concept_snr, extract_level_features

TOXICN_SEEDS = [4468, 729, 1196, 5114, 1136]
L, N_CHANNELS = 132, 3
KS = [5, 10, 20, 50]
R_TRIALS = 20      # 估计随机排序期望效果的随机试验次数
N_PERM = 2000      # 配对置换检验（符号翻转）次数
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "result"


def load_types(config):
    with open(config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json",
              encoding="utf-8") as f:
        return [i["type"] for i in json.load(f)]


def resolve_model_dir(exp_dir, seed, dataset):
    d1 = exp_dir / f"seed_{seed}"
    if (d1 / "best_model.pth").exists():
        return d1
    cands = sorted(exp_dir.glob(f"*_{dataset}_acv_s{seed}"))
    cands = [c for c in cands if (c / "best_model.pth").exists()]
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"未找到 seed={seed} 模型目录：exp_dir={exp_dir}。")


def load_models(seeds, types, config, exp_dir, dataset):
    models = []
    for seed in seeds:
        d = resolve_model_dir(exp_dir, seed, dataset)
        sd = torch.load(d / "best_model.pth", map_location="cpu")
        m = GatedConceptClassifier(L, types, config.dropout_rate,
                                   config.hidden_features, 0, N_CHANNELS)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def load_features(config, types, dataset, model_name):
    base = config.processed_path / dataset / model_name
    with open(base / f"concept_train_{model_name}_v2_3level.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(base / f"concept_test_{model_name}_v2_3level.json", encoding="utf-8") as f:
        test_data = json.load(f)
    snr = compute_concept_snr(train_data, L)
    _, _ = extract_level_features(train_data, types, snr, "full", True)
    te_X, te_y = extract_level_features(test_data, types, snr, "full", True)
    return {"snr": snr, "te_X": te_X, "te_y": te_y}


def ensemble_saliency(models, te_X):
    """梯度×输入归因（多 seed 平均）→ 每概念重要性 (n, L)。"""
    sals = []
    for m in models:
        with torch.no_grad():
            gate = torch.sigmoid(m.gate_layer(te_X))
            xg = (te_X * gate).detach()
        xg_leaf = xg.clone().requires_grad_(True)
        h = torch.relu(m.fc1(xg_leaf))
        logit = m.fc2(h)
        pt = torch.softmax(logit, dim=1)[:, 1]
        att = torch.autograd.grad(pt.sum(), xg_leaf)[0]
        sals.append((xg_leaf * att).detach())
    sal = torch.stack(sals, dim=0).mean(dim=0)          # (n, 396)
    # 通道布局为分块拼接 [p^{(3)}; p^{(2)}; d̃]，概念 i 的 3 通道在列 {i, i+L, i+2L}。
    return sal.view(-1, N_CHANNELS, L).abs().sum(dim=1).numpy()  # (n, L)


def predict_masked(models, te_X, channel_mask):
    """在门控后瓶颈表示 x̃ 上按 channel_mask 干预，返回多 seed 集成 p_toxic (n,)。"""
    probs = []
    for m in models:
        with torch.no_grad():
            gate = torch.sigmoid(m.gate_layer(te_X))
            xg = te_X * gate
            xg_masked = xg.clone()
            xg_masked[~channel_mask] = 0.0
            h = torch.relu(m.fc1(xg_masked))
            logit = m.fc2(h)
            probs.append(torch.softmax(logit, dim=1)[:, 1])
    return torch.stack(probs, dim=0).mean(dim=0).numpy()


def concept_to_channel_mask(cmask):
    """将 (n, L) 概念布尔掩码复制到 3 通道，得到 (n, 3L) 通道掩码。"""
    return np.concatenate([cmask, cmask, cmask], axis=1)


def paired_perm_test(delta, n_perm, rng, alternative="greater"):
    """配对置换检验（符号翻转）：检验 mean(delta) 与 0 的差异是否显著。

    delta[i] 为第 i 条文本在 top 排序与随机排序下的指标差。零假设下 top 与随机
    等价，故 delta[i] 的符号可随机翻转构造零分布。alternative="greater" 检验
    mean(delta)>0（完备性），"less" 检验 mean(delta)<0（充分性）。
    """
    obs = float(delta.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(delta)))
    perm_stat = (delta[None, :] * signs).mean(axis=1)
    if alternative == "less":
        p = (int((perm_stat <= obs).sum()) + 1) / (n_perm + 1)
    else:
        p = (int((perm_stat >= obs).sum()) + 1) / (n_perm + 1)
    return float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--dataset", default="TOXICN")
    p.add_argument("--model-name", default="glm-4-9b-chat")
    p.add_argument("--seeds", type=str, default=None)
    args = p.parse_args()

    config = MLPConfig()
    config.dataset_name = args.dataset
    config.model_name = args.model_name
    exp_dir = Path(args.exp_dir)

    if args.seeds is not None:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    else:
        seeds = TOXICN_SEEDS if args.dataset == "TOXICN" else sorted(
            {int(d.name.rsplit("_s", 1)[1]) for d in exp_dir.glob(f"*_{args.dataset}_acv_s*")
             if (d / "best_model.pth").exists()})

    types = load_types(config)
    models = load_models(seeds, types, config, exp_dir, args.dataset)
    feat = load_features(config, types, args.dataset, args.model_name)
    te_X = feat["te_X"]
    n = te_X.shape[0]
    print(f">>> dataset={args.dataset} seeds={seeds} n_test={n} "
          f"特征维={te_X.shape[1]}")

    saliency = ensemble_saliency(models, te_X)          # (n, L)
    full_mask = np.ones((n, L), dtype=bool)
    p_full = predict_masked(models, te_X, channel_mask=concept_to_channel_mask(full_mask))
    topk_ids = {k: [np.argsort(-saliency[i])[:k] for i in range(n)] for k in KS}

    # ===== top 排序的逐样本完备性 / 充分性 =====
    drop_topk, gap_topk = {}, {}
    for k in KS:
        cmask_rm = np.ones((n, L), dtype=bool)
        cmask_keep = np.zeros((n, L), dtype=bool)
        for i, ids in enumerate(topk_ids[k]):
            cmask_rm[i, ids] = False
            cmask_keep[i, ids] = True
        p_abl = predict_masked(models, te_X, channel_mask=concept_to_channel_mask(cmask_rm))
        p_only = predict_masked(models, te_X, channel_mask=concept_to_channel_mask(cmask_keep))
        drop_topk[k] = p_full - p_abl                  # (n,)
        gap_topk[k] = p_full - p_only                  # (n,)

    # ===== 随机排序对照 + 配对置换检验 =====
    rng = np.random.default_rng(2024)
    comp = {"topk": {}, "random": {}, "pvalue": {}}
    suff = {"topk": {}, "random": {}, "pvalue": {}}
    for k in KS:
        drop_rand_list, gap_rand_list = [], []
        for _ in tqdm(range(R_TRIALS), desc=f"random k={k}"):
            rm_ids = np.array([rng.choice(L, k, replace=False) for _ in range(n)])  # (n,k)
            cmask_rm = np.ones((n, L), dtype=bool)
            cmask_keep = np.zeros((n, L), dtype=bool)
            for i in range(n):
                cmask_rm[i, rm_ids[i]] = False
                cmask_keep[i, rm_ids[i]] = True
            p_rm = predict_masked(models, te_X, channel_mask=concept_to_channel_mask(cmask_rm))
            p_keep = predict_masked(models, te_X, channel_mask=concept_to_channel_mask(cmask_keep))
            drop_rand_list.append(p_full - p_rm)
            gap_rand_list.append(p_full - p_keep)
        drop_rand = np.array(drop_rand_list).mean(axis=0)   # (n,)
        gap_rand = np.array(gap_rand_list).mean(axis=0)     # (n,)

        comp["topk"][k] = float(drop_topk[k].mean())
        comp["random"][k] = float(drop_rand.mean())
        comp["pvalue"][k] = paired_perm_test(drop_topk[k] - drop_rand, N_PERM, rng, "greater")

        suff["topk"][k] = float(gap_topk[k].mean())
        suff["random"][k] = float(gap_rand.mean())
        suff["pvalue"][k] = paired_perm_test(gap_topk[k] - gap_rand, N_PERM, rng, "less")

    out = {
        "method": "concept_faithfulness_comprehensiveness_sufficiency",
        "reference": "DeYoung et al., 2020 (ERASER): comprehensiveness & sufficiency",
        "ranking": "gradient x input attribution at gated bottleneck x = x * sigmoid(gate(x))",
        "intervention": "zero-out concepts on gated bottleneck x (CBM bottleneck intervention)",
        "significance": "paired permutation test (top rank vs random rank), one-sided",
        "dataset": args.dataset, "model": args.model_name, "seeds": seeds,
        "n_test": n, "n_concepts": L, "concept_counts": KS,
        "random_trials": R_TRIALS, "n_perm": N_PERM,
        "metric": "Comprehensiveness: drop in p_toxic; Sufficiency: gap in p_toxic",
        "comprehensiveness": comp,
        "sufficiency": suff,
    }

    print("\n===== 完备性 Comprehensiveness（移除 top 概念后 p_toxic 下降，越大越关键）=====")
    print(f"{'概念数':>5} | {'top排序':>8} | {'随机对照':>8} | {'p值(单侧)':>10}")
    for k in KS:
        print(f"{k:>5} | {comp['topk'][k]:>8.4f} | {comp['random'][k]:>8.4f} | "
              f"{comp['pvalue'][k]:>10.5f}")

    print("\n===== 充分性 Sufficiency（仅保留 top 概念后 p_toxic 差距，越小越充分）=====")
    print(f"{'概念数':>5} | {'top排序':>8} | {'随机对照':>8} | {'p值(单侧)':>10}")
    for k in KS:
        print(f"{k:>5} | {suff['topk'][k]:>8.4f} | {suff['random'][k]:>8.4f} | "
              f"{suff['pvalue'][k]:>10.5f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / "concept_faithfulness.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_json}")


if __name__ == "__main__":
    main()