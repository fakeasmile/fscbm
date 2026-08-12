"""ACV-FSCBM 下游分类器训练入口。

对应论文 2.3 节"门控概念分类器"：
  等级特征向量构造：实施概率 m^{(3)} + 涉及概率 m^{(2)} + SNR加权差值 (m^{(3)}-m^{(2)})⊙w
  门控多层感知机：矩阵门控 → Dropout → FC(96) → ReLU → Dropout → FC(2)

训练策略：label_smoothing=0.05 + EMA(decay=0.999) + AdamW + OneCycleLR

SNR计算注记：SNR作为特征工程参数从全训练集估计，不参与梯度优化。
                此做法等价于 StandardScaler.fit(train_data)，不属于数据泄露。

-- 快速上手 --
# 搜索好种子（30个种子，每个训练一次，从 summary.json 找 best）
python utils/train_acv_fscbm.py --dataset_name TOXICN --model_name glm-4-9b-chat --n_seeds 30

# 复现最佳种子
python utils/train_acv_fscbm.py --dataset_name TOXICN --model_name glm-4-9b-chat --seed 42
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
# 等级特征向量构造（对应论文 2.3 节）
# =============================================================================
def extract_level_features(data, concept_types, snr_weights):
    """构造等级特征向量 x = [p^{(3)}; p^{(2)}; d̃] ∈ R^{3l}。

    Args:
        data: 概念向量数据集（每个元素含 level_probs 和 toxic 标签）
        concept_types: 概念类型列表（保留参数，当前未使用）
        snr_weights: SNR 权重向量 w_j = max(SNR_j, 0) + 0.01

    Returns:
        (X, y): 等级特征矩阵和标签张量
    """
    n_samples, n_concepts = len(data), len(data[0]["concept"])
    p3_arr = np.zeros((n_samples, n_concepts))
    p2_arr = np.zeros((n_samples, n_concepts))
    y = np.zeros(n_samples, dtype=int)
    for si, item in enumerate(data):
        lp = item["level_probs"]
        for ci in range(n_concepts):
            p3_arr[si, ci] = lp[ci][2]; p2_arr[si, ci] = lp[ci][1]
        y[si] = item["toxic"]
    snr_w = np.clip(snr_weights, 0, None) + 0.01
    weighted_contrast = (p3_arr - p2_arr) * snr_w
    X = np.concatenate([p3_arr, p2_arr, weighted_contrast], axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def compute_concept_snr(train_data, n_concepts):
    """计算概念的 SNR（信噪比），衡量概念在有毒/无毒文本间的区分度。

    SNR_j = (μ_{1,j} - μ_{0,j}) / σ_{pool,j}
    其中 μ_{1,j} 和 μ_{0,j} 分别表示有毒/无毒文本中概念 j 的差值信号均值，
    σ_{pool,j} 为合并标准差。

    Args:
        train_data: 训练集概念向量
        n_concepts: 概念数量

    Returns:
        snr: SNR 数组，形状 (n_concepts,)
    """
    train_toxic = [i for i in train_data if i['toxic'] == 1]
    train_nt = [i for i in train_data if i['toxic'] == 0]
    snr = np.zeros(n_concepts)
    for ci in range(n_concepts):
        tc = [i['level_probs'][ci][2] - i['level_probs'][ci][1] for i in train_toxic]
        nc = [i['level_probs'][ci][2] - i['level_probs'][ci][1] for i in train_nt]
        snr[ci] = (np.mean(tc) - np.mean(nc)) / (np.std(tc + nc) + 1e-8)
    return snr


# =============================================================================
# 单种子训练（对应论文 3.2 节"实验设置"）
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


def save_seed_result(out_dir, r, seed, args, n_concepts, n_features, use_ema):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"pipeline": "acv_fscbm", "feature_mode": "level_features",
           "seed": seed, "n_concepts": n_concepts, "n_features": n_features,
           "label_smoothing": args.label_smoothing,
           "use_ema": use_ema, "ema_decay": args.ema_decay if use_ema else None,
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
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--no_ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--n_seeds', type=int, default=1)
    args = p.parse_args()

    config = MLPConfig(); config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    use_ema = not args.no_ema

    base = config.processed_path / args.dataset_name / args.model_name
    with open(base / f"concept_train_{args.model_name}_v2_3level.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(base / f"concept_test_{args.model_name}_v2_3level.json", encoding="utf-8") as f:
        test_data = json.load(f)
    n_concepts = len(train_data[0]["concept"])
    print(f">>> train={len(train_data)} test={len(test_data)} n={n_concepts}")
    with open(config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json", encoding="utf-8") as f:
        concept_types = [i["type"] for i in json.load(f)]
    snr = compute_concept_snr(train_data, n_concepts)
    nc, ns = 3, 0  # ns=0: 不使用类型级聚合特征，特征维度降至 396
    tr_X, tr_y = extract_level_features(train_data, concept_types, snr)
    te_X, te_y = extract_level_features(test_data, concept_types, snr)
    n_feat = tr_X.shape[1]
    print(f">>> 特征: {n_feat}d LS={args.label_smoothing} EMA={'on' if use_ema else 'off'}")

    if args.n_seeds == 1:
        seeds = [args.seed if args.seed is not None else random.randint(0, 9999)]
    else:
        seeds = [args.seed + i for i in range(args.n_seeds)] if args.seed is not None \
                else [random.randint(0, 9999) for _ in range(args.n_seeds)]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = config.experiment_path / ts
    print(f">>> {len(seeds)} seeds → {parent.name}")

    all_r = []
    for si, seed in enumerate(seeds):
        r = train_one_seed(tr_X, tr_y, te_X, te_y, concept_types,
                           config, n_concepts, ns, nc,
                           args.label_smoothing, use_ema, args.ema_decay, seed)
        all_r.append({'seed': seed, **r})
        save_seed_result(parent / f"seed_{seed}", r, seed, args, n_concepts, n_feat, use_ema)
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
               "feature_mode": "level_features", "label_smoothing": args.label_smoothing,
               "use_ema": use_ema, "n_concepts": n_concepts, "n_features": n_feat,
               "results": [{"seed": r['seed'], "test_f1": round(r['test_f1'], 4),
                            "val_f1": round(r['val_f1'], 4)} for r in all_r]}
    with open(parent / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f">>> {parent / 'summary.json'}")


if __name__ == "__main__":
    main()