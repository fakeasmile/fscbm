# -*- coding: utf-8 -*-
"""概念层可解释性定量评估 · 可预测性（predictability）—— 对应论文 2.3.4 节「可预测性评估」，产出表6。

【CBM 标准做法】本脚本验证 CBM 三大验证目标之一的「可预测性（predictability）」
（Koh et al., 2020, ICML；Margeloiu et al., 2021 Do Concept Bottleneck Models Learn
As Intended?)：仅凭概念层特征即可预测目标标签，预测越强说明概念层承载的分类信息越充分。
实现：只用手工概念特征 [p^{(3)}; p^{(2)}; SNR保号加权差值]（396 维）训练线性逻辑回归探针
（linear probe），在测试集上报告 AUC 与加权 F1。另设两组对照排除虚假判别：①打乱训练集标签，
排除过拟合；②以同维随机高斯特征替换概念特征，排除维度冗余。

【需要的数据/目录】（均在项目根目录 D:/fscbm 下，须已离线生成完毕，运行时不调用 LLM）：
  1. LLM 三级评分概率（每样本 132 概念 × 等级 {1,2,3} 的概率，训练/测试各一份）：
     D:/fscbm/data/processed/TOXICN/glm-4-9b-chat/concept_train_glm-4-9b-chat_v2_3level.json
     D:/fscbm/data/processed/TOXICN/glm-4-9b-chat/concept_test_glm-4-9b-chat_v2_3level.json
  2. 形容词类型定义（132 个形容词的类型标记）：
     D:/fscbm/data/raw/adjective/toxic_adjectives_v2_types.json
  3. 已训练门控分类器权重（5 个 seed 集成，逐 seed 读 best_model.pth）：
     D:/fscbm/experiments/TOXICN(acv_fscbm)/<时间戳>_TOXICN_acv_s{seed}/best_model.pth
     （seed ∈ {4468, 729, 1196, 5114, 1136}，由 resolve_model_dir 自动匹配“时间戳”目录）
  4. 毒性标签（0/1）随概念 json 一并读取（tr_y / te_y），无需额外文件。

【不需要的数据】不读取任何「文本-概念等级」的人工标注，尤其不读
annotation/concept_fidelity_manual.csv；概念评分由 LLM 自动生成，契合「免概念标注 CBM」范式。

【附加产出】同一脚本还计算「门控贡献集中度」（top-k 累积贡献占比、有效概念数 NEC），
作为概念判别集中度的辅助量化（当前论文表6未引用，仅作备查）。

【用法】在 D:/fscbm 下，toxicnew 环境：
  python xai/concept_informativeness.py --exp-dir "experiments/TOXICN(acv_fscbm)" --dataset TOXICN

【输出】控制台 + xai/result/concept_informativeness.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("d:/fscbm")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.MLP_config import MLPConfig
from models.gated_classifier import GatedConceptClassifier
from utils.train_acv_fscbm import compute_concept_snr, extract_level_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

SEEDS = [4468, 729, 1196, 5114, 1136]
L, N_CHANNELS = 132, 3
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "result"


def load_types(config):
    import json as _json
    with open(config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json",
              encoding="utf-8") as f:
        return [i["type"] for i in _json.load(f)]


def resolve_model_dir(exp_dir, seed, dataset):
    d1 = exp_dir / f"seed_{seed}"
    if (d1 / "best_model.pth").exists():
        return d1
    cands = sorted(exp_dir.glob(f"*_{dataset}_acv_s{seed}"))
    cands = [c for c in cands if (c / "best_model.pth").exists()]
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"未找到 seed={seed} 模型目录: {exp_dir}")


def load_features(config, types, dataset, model_name):
    import json as _json
    base = config.processed_path / dataset / model_name
    with open(base / f"concept_train_{model_name}_v2_3level.json", encoding="utf-8") as f:
        train_data = _json.load(f)
    with open(base / f"concept_test_{model_name}_v2_3level.json", encoding="utf-8") as f:
        test_data = _json.load(f)
    snr = compute_concept_snr(train_data, L)
    tr_X, tr_y = extract_level_features(train_data, types, snr, "full", True)
    te_X, te_y = extract_level_features(test_data, types, snr, "full", True)
    return {"snr": snr, "tr_X": tr_X, "tr_y": tr_y, "te_X": te_X, "te_y": te_y}


def gate_contrib(config, types, exp_dir, te_X, dataset):
    su = torch.zeros_like(te_X)
    for seed in SEEDS:
        d = resolve_model_dir(exp_dir, seed, dataset)
        sd = torch.load(d / "best_model.pth", map_location="cpu")
        m = GatedConceptClassifier(L, types, config.dropout_rate,
                                   config.hidden_features, 0, N_CHANNELS)
        m.load_state_dict(sd); m.eval()
        with torch.no_grad():
            gate = torch.sigmoid(m.gate_layer(te_X))
        su += te_X * gate
    xtilde = su / len(SEEDS)
    return xtilde.view(-1, L, N_CHANNELS).abs().amax(dim=2).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--dataset", default="TOXICN")
    p.add_argument("--model-name", default="glm-4-9b-chat")
    p.add_argument("--seed", type=int, default=42, help="逻辑回归打乱对照的随机种子")
    args = p.parse_args()

    config = MLPConfig()
    config.dataset_name = args.dataset
    config.model_name = args.model_name
    exp_dir = Path(args.exp_dir)
    types = load_types(config)
    feat = load_features(config, types, args.dataset, args.model_name)
    tr_X, tr_y = feat["tr_X"].numpy(), feat["tr_y"].numpy()
    te_X, te_y = feat["te_X"].numpy(), feat["te_y"].numpy()

    # ===== 1. 概念层判别信息量（predictability）=====
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(tr_X, tr_y)
    prob = clf.predict_proba(te_X)[:, 1]
    pred = clf.predict(te_X)
    pred_info = {
        "auc": float(roc_auc_score(te_y, prob)),
        "f1_weighted": float(f1_score(te_y, pred, average="weighted")),
    }

    # 对照1：打乱标签（排除过拟合——若分类器无病，打乱标签后 AUC 应≈0.5）
    rng = np.random.default_rng(args.seed)
    tr_y_shuf = tr_y[rng.permutation(len(tr_y))].copy()
    clf_lab = LogisticRegression(C=1.0, max_iter=2000)
    clf_lab.fit(tr_X, tr_y_shuf)
    prob_lab = clf_lab.predict_proba(te_X)[:, 1]
    pred_lab = clf_lab.predict(te_X)
    label_perm_info = {
        "auc": float(roc_auc_score(te_y, prob_lab)),
        "f1_weighted": float(f1_score(te_y, pred_lab, average="weighted")),
    }

    # 对照2：随机高斯特征（同维度，排除"仅靠维度多"的平凡结果）
    rng2 = np.random.default_rng(args.seed + 1)
    tr_rand = rng2.normal(size=tr_X.shape)
    te_rand = rng2.normal(size=te_X.shape)
    clf_rand = LogisticRegression(C=1.0, max_iter=2000)
    clf_rand.fit(tr_rand, tr_y)
    prob_rand = clf_rand.predict_proba(te_rand)[:, 1]
    pred_rand = clf_rand.predict(te_rand)
    random_feat_info = {
        "auc": float(roc_auc_score(te_y, prob_rand)),
        "f1_weighted": float(f1_score(te_y, pred_rand, average="weighted")),
    }

    # ===== 2. 门控贡献集中度（interpretability）=====
    contrib = gate_contrib(config, types, exp_dir, feat["te_X"], args.dataset)  # (n_test, L)
    per_concept = contrib.mean(axis=0)                    # 跨样本平均贡献 (L,)
    order = np.argsort(-per_concept)                      # 降序
    total = per_concept.sum() + 1e-12
    topk = {}
    for k in (5, 10, 20, 50):
        topk[f"top_{k}"] = float(per_concept[order[:k]].sum() / total)
    # 有效概念数 NEC = (Σc)^2 / Σc^2（Herfindahl 倒数）
    nec = float((per_concept.sum() ** 2) / (per_concept ** 2).sum())
    n_nonzero = int((per_concept > 1e-8).sum())

    out = {
        "method": "concept_informativeness",
        "predictability": {"concept_features": pred_info,
                           "label_permutation": label_perm_info,
                           "random_features": random_feat_info},
        "interpretability": {
            "concept_cumulative_share": topk,
            "NEC_effective_concepts": nec,
            "n_nonzero_concepts": n_nonzero,
            "total_concepts": L,
        },
    }

    print("\n===== 概念层判别信息量（逻辑回归，396 维概念特征）=====")
    print(f"概念特征 : AUC={pred_info['auc']:.4f}  weightedF1={pred_info['f1_weighted']:.4f}")
    print(f"打乱标签 : AUC={label_perm_info['auc']:.4f}  weightedF1={label_perm_info['f1_weighted']:.4f}")
    print(f"随机特征 : AUC={random_feat_info['auc']:.4f}  weightedF1={random_feat_info['f1_weighted']:.4f}")
    print(f"\n===== 门控贡献集中度 =====")
    for k, v in topk.items():
        print(f"{k} 概念累积贡献占比: {v:.3f}")
    print(f"有效概念数 NEC: {nec:.1f} / {L}   非零概念: {n_nonzero}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / "concept_informativeness.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_json}")


if __name__ == "__main__":
    main()