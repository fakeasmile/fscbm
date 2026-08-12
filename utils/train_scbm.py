"""SCBM 基线分类器训练入口。

对应论文 3.1 节"基线方法"中的 SCBM 迁移：
  - 采用与本文相同的形容词词典（v2 词典，132 个形容词）
  - 保留原始二元概念评估方式（相关/不相关）
  - 输入特征：132 维二元概念分数（P("是")）
  - 分类器：与 ACV-FSCBM 一致的门控 MLP

训练策略：label_smoothing=0.05 + EMA(decay=0.999) + AdamW + OneCycleLR

-- 快速上手 --
# 固定种子全量实验
python utils/train_scbm.py --dataset_name TOXICN --model_name glm-4-9b-chat --seed 42

-- 参数说明 --
--dataset_name    数据集名称，如 TOXICN
--model_name      LLM 概念向量模型名，如 glm-4-9b-chat
--seed            固定随机种子（不指定时随机生成）
--n_seeds         搜索的种子数量（>1 时批量训练，从 summary.json 找最优）

注：label_smoothing / use_ema / ema_decay 为训练超参数，在 configs/MLP_config.py 中配置。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime

import matplotlib; matplotlib.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from configs.MLP_config import MLPConfig
from models.gated_classifier import GatedConceptClassifier


# =============================================================================
# 随机种子控制
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


# =============================================================================
# 特征提取（SCBM 二元概念分数）
# =============================================================================
def extract_scbm_features(data):
    """提取 SCBM 二元概念分数。

    SCBM 使用二元概念评估，每概念输出一个 P("是") 分数，
    特征向量 x ∈ R^{l}，其中 l = 132。

    Args:
        data: SCBM 二元概念向量数据集（每个元素含 concept 和 toxic 标签）

    Returns:
        (X, y): 特征矩阵和标签张量
    """
    n_samples, n_concepts = len(data), len(data[0]["concept"])
    X = np.zeros((n_samples, n_concepts))
    y = np.zeros(n_samples, dtype=int)
    for si, item in enumerate(data):
        X[si, :] = item["concept"]
        y[si] = item["toxic"]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =============================================================================
# 单种子训练
# =============================================================================
def train_one_seed(train_X, train_y, test_X, test_y, concept_types,
                   config, n_concepts, n_summary, n_main_channels,
                   label_smoothing, use_ema, ema_decay, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_X, va_X, tr_y, va_y, _, _ = train_test_split(
        train_X.numpy(), train_y.numpy(), np.arange(len(train_X)),
        test_size=0.2, stratify=train_y.numpy(), random_state=seed)
    tr_X, va_X = torch.tensor(tr_X), torch.tensor(va_X)
    tr_y, va_y = torch.tensor(tr_y), torch.tensor(va_y)
    train_loader = DataLoader(TensorDataset(tr_X, tr_y), batch_size=config.batch_size,
                              shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(TensorDataset(va_X, va_y), batch_size=config.batch_size)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=config.batch_size)

    model = GatedConceptClassifier(
        n_concepts, concept_types, config.dropout_rate, config.hidden_features,
        n_summary, n_main_channels).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)
    steps = len(train_loader) * config.epochs
    sch = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.max_lr, total_steps=steps,
        pct_start=config.pct_start, anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor, final_div_factor=config.final_div_factor)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)) if use_ema else None

    best_val, best_sd, best_ep, no_imp = 0.0, None, 0, 0
    hist_v, hist_t = [], []
    pbar = tqdm(range(config.epochs), desc=f"Seed {seed}")
    for ep in pbar:
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = criterion(model(bx), by)
            loss.backward(); opt.step(); sch.step()
            if ema is not None: ema.update_parameters(model)
        ev = ema if ema is not None else model
        ev.eval()
        vp, vl = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                vp.extend(torch.argmax(ev(bx.to(device)), 1).cpu().numpy())
                vl.extend(by.numpy())
        vf = f1_score(vl, vp, average='weighted')
        tp, tl = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                tp.extend(torch.argmax(ev(bx.to(device)), 1).cpu().numpy())
                tl.extend(by.numpy())
        tf_ = f1_score(tl, tp, average='weighted')
        hist_v.append(vf); hist_t.append(tf_)
        pbar.set_postfix({'val': f'{vf:.4f}', 'test': f'{tf_:.4f}', 'best': f'{best_val:.4f}'})
        if vf > best_val:
            best_val = vf; best_ep = ep + 1; no_imp = 0
            src = ema.module if ema is not None else model
            best_sd = {k: v.clone() for k, v in src.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= config.patience:
            pbar.close(); break

    model.load_state_dict(best_sd); model.eval()
    ap, al = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            ap.extend(torch.argmax(model(bx.to(device)), 1).cpu().numpy())
            al.extend(by.numpy())
    tf = f1_score(al, ap, average='weighted')
    tp = precision_score(al, ap, average='macro', zero_division=0)
    tr = recall_score(al, ap, average='macro', zero_division=0)
    nr = recall_score(al, ap, labels=[0], average=None)[0]
    xr = recall_score(al, ap, labels=[1], average=None)[0]
    cr = classification_report(al, ap, target_names=["Non-Toxic", "Toxic"])
    return {'val_f1': best_val, 'test_f1': tf, 'precision': tp, 'recall': tr,
            'nt_recall': nr, 'tx_recall': xr, 'best_epoch': best_ep,
            'report': cr, 'hist_v': hist_v, 'hist_t': hist_t, 'state_dict': best_sd}


# =============================================================================
# 结果保存与可视化
# =============================================================================
def plot_metrics(out_dir, hist_v, hist_t):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_v, label='Val F1', color='tab:blue')
    ax.plot(hist_t, label='Test F1 (obs)', color='tab:red', linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted F1')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir / 'metrics.png'); plt.close()


def save_seed_result(out_dir, r, seed, config, args, n_concepts, n_features):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"pipeline": "scbm", "feature_mode": "binary",
           "seed": seed, "n_concepts": n_concepts, "n_features": n_features,
           "label_smoothing": config.label_smoothing,
           "use_ema": config.use_ema, "ema_decay": config.ema_decay if config.use_ema else None,
           "val_f1": round(r['val_f1'], 4), "test_f1": round(r['test_f1'], 4)}
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    torch.save(r['state_dict'], out_dir / "best_model.pth")
    plot_metrics(out_dir, r['hist_v'], r['hist_t'])
    trd = out_dir / "test_results"; trd.mkdir(exist_ok=True)
    with open(trd / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Seed={seed} Val={r['val_f1']:.4f} Test={r['test_f1']:.4f}\n")
        f.write("-" * 40 + "\n" + r['report'])


# =============================================================================
# 主入口
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--n_seeds', type=int, default=1)
    args = p.parse_args()

    config = MLPConfig(); config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    use_ema = config.use_ema

    base = config.processed_path / args.dataset_name / args.model_name
    with open(base / f"concept_train_{args.model_name}_scbm.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(base / f"concept_test_{args.model_name}_scbm.json", encoding="utf-8") as f:
        test_data = json.load(f)
    n_concepts = len(train_data[0]["concept"])
    print(f">>> train={len(train_data)} test={len(test_data)} n={n_concepts}")
    with open(config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json", encoding="utf-8") as f:
        concept_types = [i["type"] for i in json.load(f)]

    # SCBM 特征：l 维二元概念分数（无 SNR 加权，无差值信号）
    tr_X, tr_y = extract_scbm_features(train_data)
    te_X, te_y = extract_scbm_features(test_data)
    n_feat = tr_X.shape[1]
    print(f">>> SCBM 特征: {n_feat}d LS={config.label_smoothing} EMA={'on' if use_ema else 'off'}")

    if args.n_seeds == 1:
        seeds = [args.seed if args.seed is not None else random.randint(0, 9999)]
    else:
        seeds = [args.seed + i for i in range(args.n_seeds)] if args.seed is not None \
                else [random.randint(0, 9999) for _ in range(args.n_seeds)]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = config.experiment_path / f"{ts}_scbm"
    print(f">>> {len(seeds)} seeds → {parent.name}")

    all_r = []
    for si, seed in enumerate(seeds):
        r = train_one_seed(tr_X, tr_y, te_X, te_y, concept_types,
                           config, n_concepts, 0, 1,  # ns=0, nc=1
                           config.label_smoothing, use_ema, config.ema_decay, seed)
        all_r.append({'seed': seed, **r})
        save_seed_result(parent / f"seed_{seed}", r, seed, config, args, n_concepts, n_feat)
        print(f"  seed={seed}: val={r['val_f1']:.4f} test={r['test_f1']:.4f} ep={r['best_epoch']}")

    all_r.sort(key=lambda x: x['test_f1'], reverse=True)
    print(f"\n{'='*50}")
    print(f"  {'Seed':<8}{'Val':<10}{'Test':<10}{'NT':<10}{'TX':<10}")
    print(f"  {'-'*46}")
    for r in all_r[:20]:
        print(f"  {r['seed']:<8}{r['val_f1']:<10.4f}{r['test_f1']:<10.4f}"
              f"{r['nt_recall']:<10.4f}{r['tx_recall']:<10.4f}")
    best = all_r[0]
    print(f"\n>>> 最佳 seed={best['seed']} F1={best['test_f1']:.4f}")
    print(f">>> 复现: --seed {best['seed']}")

    summary = {"timestamp": ts, "n_seeds": len(seeds),
               "feature_mode": "scbm_binary", "label_smoothing": config.label_smoothing,
               "use_ema": use_ema, "n_concepts": n_concepts, "n_features": n_feat,
               "results": [{"seed": r['seed'], "test_f1": round(r['test_f1'], 4),
                            "val_f1": round(r['val_f1'], 4)} for r in all_r]}
    with open(parent / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f">>> {parent / 'summary.json'}")


if __name__ == "__main__":
    main()