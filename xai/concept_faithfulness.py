# -*- coding: utf-8 -*-
"""概念决策忠实度评估（Comprehensiveness & Sufficiency）—— 对应论文 2.3.4 节「决策忠实度评估」，产出表7。

【CBM / ERASER 标准做法】采用 ERASER（DeYoung et al., 2020, ACL）的完备性/充分性指标，
把 token 级 rationale 迁移到概念层：概念即概念瓶颈层的「软 rationale」。
ERASER 原始定义中，j 取「模型对该样本的预测类」ŷ（原始论文与后续复现工作一致，如 Yu et al. 2019、
“Evaluating and Characterizing Human Rationales”, EMNLP 2020）：

    完备性 Comprehensiveness = m(x)_ŷ − m(x \\ r)_ŷ   移除 rationale 后预测类置信度的下降（越大越忠实）
    充分性 Sufficiency         = m(x)_ŷ − m(r)_ŷ       仅保留 rationale 后预测类置信度与原决策的差距（越小越忠实）

即：**度量必须相对于「被解释的那个决策」进行**。本文解释的是「判为有毒」这一决策，故方向固定为有毒类
（p_toxic = softmax(logits)[:,1]），评估范围也必须落在决策为「有毒」的样本上。

【评估范围 scope —— 这是表7数值口径的唯一变量】
    gold_toxic（默认）：测试集中真实有毒样本。决策对象 = 有毒类，与解释方向一致，且不以模型自身预测挑样本。
    pred_toxic        ：模型预测为有毒的样本（ŷ=1）。最贴合 ERASER 「j = 预测类」的字面定义。
    all               ：全部测试样本。含模型判为无毒的样本，其决策对象并非有毒类，
                        按有毒类概率取均值会与解释方向失配并稀释效应（不推荐，仅作对照）。
  三个口径共用同一套逐样本归因 / 干预 / 随机对照结果，仅取子集不同，**一次运行同时给出**，
  正常情形下三者的方向与显著性结论应当完全一致。

概念瓶颈：x = [p^{(3)}; p^{(2)}; d̃] ∈ R^{3L}，门控后表示 x̃ = x ⊙ σ(gate(x)) 即下游分类器实际使用的瓶颈层。
概念 c_i 的 3 个通道位于列 {c_i, c_i+L, c_i+2L}。

指标：
  1. 完备性：移除按「梯度×输入归因」降序排列的 top 概念后 p_toxic 的下降，越大说明被移除概念越关键。
  2. 充分性：仅保留 top 概念后 p_toxic 与完整输入的差距，越小越充分。
  3. 显著性检验：配对置换检验（逐样本 top 排序 vs 随机排序的效果差），单侧 p 值。

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
  可选：--scope {gold_toxic,pred_toxic,all}  指定写入 JSON 顶层的主口径（默认 gold_toxic）。

【输出】控制台（三个口径 + 主口径重排表）+ xai/result/concept_faithfulness.json
        （顶层 comprehensiveness/sufficiency 为主口径；scopes 下记录全部口径，便于复核）
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
SCOPES = ["gold_toxic", "pred_toxic", "all"]
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


def scope_masks(p_full, te_y):
    """三个评估范围的布尔掩码（ERASER 语义：只度量「决策为有毒」的样本）。"""
    y = te_y.numpy().astype(int)
    return {
        "gold_toxic": (y == 1),            # 真实有毒：决策对象与解释方向一致
        "pred_toxic": (p_full >= 0.5),     # 模型预测有毒：等价于 ERASER 的 j = 预测类
        "all": np.ones_like(y, dtype=bool),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--dataset", default="TOXICN")
    p.add_argument("--model-name", default="glm-4-9b-chat")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--scope", type=str, default="gold_toxic", choices=SCOPES,
                   help="写入 JSON 顶层的主评估范围（默认 gold_toxic）")
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
    p_full_mean = float(p_full.mean())

    masks = scope_masks(p_full, feat["te_y"])
    for name in SCOPES:
        mk = masks[name]
        print(f">>> scope={name:<11} n={int(mk.sum()):>5} "
              f"baseline p_toxic={float(p_full[mk].mean()):.4f}")
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
    # 【随机数消耗顺序说明】每个概念数按「先抽 20 组随机概念 → 再做主口径的两项置换检验」的顺序消耗
    # 主随机流，与既有实现完全一致，保证主口径数值（含随机对照列与 p 值）与论文表 7 逐位一致；
    # 其余口径的置换检验改用独立随机流（rng_aux），不影响主口径。
    rng = np.random.default_rng(2024)
    rng_aux = np.random.default_rng(2024)
    drop_rand, gap_rand = {}, {}
    pval_comp_primary, pval_suff_primary = {}, {}
    mk_primary = masks[args.scope]
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
        drop_rand[k] = np.array(drop_rand_list).mean(axis=0)   # (n,)
        gap_rand[k] = np.array(gap_rand_list).mean(axis=0)     # (n,)
        # 主口径的配对置换检验（沿用主随机流，顺序不可调整）
        pval_comp_primary[k] = paired_perm_test(drop_topk[k][mk_primary] - drop_rand[k][mk_primary],
                                                N_PERM, rng, "greater")
        pval_suff_primary[k] = paired_perm_test(gap_topk[k][mk_primary] - gap_rand[k][mk_primary],
                                                N_PERM, rng, "less")

    # ===== 按三个口径汇总（共用同一套逐样本结果，仅取子集） =====
    scope_results = {}
    for name in SCOPES:
        mk = masks[name]
        comp = {"topk": {}, "random": {}, "pvalue": {}}
        suff = {"topk": {}, "random": {}, "pvalue": {}}
        for k in KS:
            comp["topk"][k] = float(drop_topk[k][mk].mean())
            comp["random"][k] = float(drop_rand[k][mk].mean())
            suff["topk"][k] = float(gap_topk[k][mk].mean())
            suff["random"][k] = float(gap_rand[k][mk].mean())
            if name == args.scope:
                comp["pvalue"][k] = pval_comp_primary[k]
                suff["pvalue"][k] = pval_suff_primary[k]
            else:
                comp["pvalue"][k] = paired_perm_test(drop_topk[k][mk] - drop_rand[k][mk],
                                                     N_PERM, rng_aux, "greater")
                suff["pvalue"][k] = paired_perm_test(gap_topk[k][mk] - gap_rand[k][mk],
                                                     N_PERM, rng_aux, "less")
        scope_results[name] = {
            "n": int(mk.sum()),
            "baseline_p_toxic": float(p_full[mk].mean()),
            "comprehensiveness": comp,
            "sufficiency": suff,
        }

    primary = scope_results[args.scope]
    out = {
        "method": "concept_faithfulness_comprehensiveness_sufficiency",
        "reference": "DeYoung et al., 2020 (ERASER): comprehensiveness & sufficiency",
        "ranking": "gradient x input attribution at gated bottleneck x = x * sigmoid(gate(x))",
        "intervention": "zero-out concepts on gated bottleneck x (CBM bottleneck intervention)",
        "significance": "paired permutation test (top rank vs random rank), one-sided",
        "dataset": args.dataset, "model": args.model_name, "seeds": seeds,
        "n_test": n, "n_concepts": L, "concept_counts": KS,
        "random_trials": R_TRIALS, "n_perm": N_PERM,
        "p_full_mean": p_full_mean,
        "primary_scope": args.scope,
        "scope_note": ("ERASER 按模型预测类度量；本文解释对象为『判为有毒』的决策，"
                       "故评估范围取决策为有毒的样本（gold_toxic 默认 / pred_toxic 为字面口径），"
                       "all 含被判为无毒的样本、会稀释效应，仅作对照。"),
        "metric": "Comprehensiveness: drop in p_toxic; Sufficiency: gap in p_toxic",
        "comprehensiveness": primary["comprehensiveness"],
        "sufficiency": primary["sufficiency"],
        "scopes": scope_results,
    }

    for name in SCOPES:
        tag = "（主口径）" if name == args.scope else ""
        r = scope_results[name]
        print(f"\n########## scope={name}  n={r['n']}  baseline p_toxic={r['baseline_p_toxic']:.4f} {tag}")
        print("===== 完备性 Comprehensiveness（移除 top 概念后 p_toxic 下降，越大越关键）=====")
        print(f"{'概念数':>5} | {'top排序':>8} | {'随机对照':>8} | {'p值(单侧)':>10}")
        for k in KS:
            c = r["comprehensiveness"]
            print(f"{k:>5} | {c['topk'][k]:>8.4f} | {c['random'][k]:>8.4f} | "
                  f"{c['pvalue'][k]:>10.5f}")
        print("===== 充分性 Sufficiency（仅保留 top 概念后 p_toxic 差距，越小越充分）=====")
        print(f"{'概念数':>5} | {'top排序':>8} | {'随机对照':>8} | {'p值(单侧)':>10}")
        for k in KS:
            s = r["sufficiency"]
            print(f"{k:>5} | {s['topk'][k]:>8.4f} | {s['random'][k]:>8.4f} | "
                  f"{s['pvalue'][k]:>10.5f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / "concept_faithfulness.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_json}")


if __name__ == "__main__":
    main()
