import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, average_precision_score

from model.seq import Branch1
from model.structure import Branch2
from model.fused import FusionPredictor
import pandas as pd  # 新增



# ----------------- Dataset -----------------
class ProteinDataset(Dataset):
    def __init__(self, esm_features, t5_features, pdb_files, labels, lengths):
        self.esm_features = esm_features
        self.t5_features = t5_features
        self.pdb_files = pdb_files
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.esm_features[idx],
            self.t5_features[idx],
            self.pdb_files[idx],
            self.labels[idx],
            self.lengths[idx]
        )


# ----------------- FASTA 标签读取 -----------------
def read_labels_from_fasta(fasta_file, max_len=300):
    labels, lengths = [], []
    with open(fasta_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                if i + 2 < len(lines):
                    label_str = lines[i + 2]
                    original_length = len(label_str)
                    current_label = label_str[:max_len] if original_length > max_len else label_str.ljust(max_len, '0')
                    labels.append([int(c) for c in current_label])
                    lengths.append(min(original_length, max_len))
                i += 3
            else:
                i += 1
    return torch.tensor(labels, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int32)


# ----------------- PDB 文件排序 -----------------
def sort_pdb_files_by_number(pdb_files):
    def extract_number(filename):
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        try:
            return int(name_without_ext)
        except ValueError:
            return float('inf')
    return sorted(pdb_files, key=extract_number)


# ----------------- Focal Loss（用于测试评估） -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        inputs = torch.clamp(inputs, -10, 10)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if mask is not None:
            focal_loss = focal_loss * mask
        return focal_loss.sum() / (mask.sum() + 1e-8) if mask is not None else focal_loss.mean()


# ----------------- 测试主流程 -----------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === 数据路径 ===
    esm_path = '/home/ys/exhd2/sunhuaiyang2/predict/feature/test/test2_esm2_features.npy'
    t5_path = '/home/ys/exhd2/sunhuaiyang2/predict/feature/test/test2_T5_features.npy'
    fasta_path = '/home/ys/exhd2/sunhuaiyang2/predict/data/test2_with_title.fasta'
    pdb_dir = '/home/ys/exhd2/sunhuaiyang2/predict/data/test2/pdb_files'
    model_dir = '/home/ys/exhd2/sunhuaiyang2/predict/gai2xian_savemodel'
    csv_save_dir = './test_results'  # CSV 保存目录

    # === 加载特征 ===
    X_esm = torch.tensor(np.load(esm_path), dtype=torch.float32)
    X_t5 = torch.tensor(np.load(t5_path), dtype=torch.float32)
    labels, lengths = read_labels_from_fasta(fasta_path)

    all_pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    pdb_files = sort_pdb_files_by_number(all_pdb_files)

    # === 检查数据一致性 ===
    min_len = min(len(X_esm), len(pdb_files), len(labels))
    X_esm, X_t5, labels, lengths = X_esm[:min_len], X_t5[:min_len], labels[:min_len], lengths[:min_len]
    pdb_files = pdb_files[:min_len]

    dataset = ProteinDataset(X_esm, X_t5, pdb_files, labels, lengths)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    focal_fn = FocalLoss(alpha=2, gamma=0.5)

    results = []

    # === 遍历每个fold模型 ===
    for fold in range(1, 6):
        model_path = os.path.join(model_dir, f'fold{fold}_best.pth')
        if not os.path.exists(model_path):
            print(f"⚠️ Missing model: {model_path}")
            continue

        print(f"\n=== Testing Fold {fold} ===")

        # 加载模型
        seq_branch = Branch1().to(device)
        struct_branch = Branch2(device=device).to(device)
        fusion_module = FusionPredictor(dim=512, hidden_dim=512, out_dim=1).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        seq_branch.load_state_dict(checkpoint['seq'])
        struct_branch.load_state_dict(checkpoint['struct'])
        fusion_module.load_state_dict(checkpoint['fusion'])

        seq_branch.eval()
        struct_branch.eval()
        fusion_module.eval()

        all_labels, all_probs = [], []
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for esm_feat, t5_feat, pdb_path, targets, lens in loader:
                esm_feat = esm_feat.to(device)
                t5_feat = t5_feat.to(device)
                targets = targets.to(device)
                lens = lens.to(device)

                seq_final = seq_branch(esm_feat, t5_feat)
                struct_out = struct_branch([pdb_path[i] for i in range(len(pdb_path))])
                fused_logits = fusion_module(seq_final, struct_out, return_features=False)

                B, L = fused_logits.shape
                mask_valid = (torch.arange(L, device=device).unsqueeze(0) < lens.unsqueeze(1)).float()

                loss = focal_fn(fused_logits, targets, mask_valid)
                val_loss += loss.item() * mask_valid.sum().item()
                total_samples += mask_valid.sum().item()

                probs = torch.sigmoid(fused_logits).cpu().numpy()
                mask_np = mask_valid.cpu().numpy().astype(bool)

                for b in range(B):
                    valid_idx = mask_np[b]
                    all_labels.extend(targets[b][valid_idx].cpu().numpy().flatten())
                    all_probs.extend(probs[b][valid_idx].flatten())

        # === 保存 CSV ===
        df = pd.DataFrame({
            "label": all_labels,
            "prob": all_probs
        })
        csv_path = os.path.join(csv_save_dir, f"test1_fold{fold}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")

        # === 计算指标 ===
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 预测类别（0.5 阈值）
        preds = (all_probs >= 0.5).astype(int)

        # Accuracy
        acc = (preds == all_labels).mean()

        # F1-score
        from sklearn.metrics import f1_score, balanced_accuracy_score

        try:
            f1 = f1_score(all_labels, preds, average="binary")
        except:
            f1 = 0.0

        # Balanced Accuracy
        try:
            bacc = balanced_accuracy_score(all_labels, preds)
        except:
            bacc = 0.0

        # AUC, AP, MCC
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, preds)

        print(
            f"Fold {fold} Results — "
            f"AUC={auc:.4f}, AP={ap:.4f}, ACC={acc:.4f}, "
            f"BACC={bacc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}"
        )

        results.append({
            "fold": fold,
            "AUC": auc,
            "AP": ap,
            "ACC": acc,
            "BACC": bacc,
            "F1": f1,
            "MCC": mcc
        })

    # === 汇总平均指标 ===
    # === 汇总平均指标 ===
    print("\n=== Final Test Results ===")
    if results:
        avg_metrics = {
            k: np.mean([r[k] for r in results])
            for k in results[0].keys() if k != "fold"
        }

        print(
            f"AUC={avg_metrics['AUC']:.4f}, "
            f"AP={avg_metrics['AP']:.4f}, "
            f"ACC={avg_metrics['ACC']:.4f}, "
            f"BACC={avg_metrics['BACC']:.4f}, "
            f"F1={avg_metrics['F1']:.4f}, "
            f"MCC={avg_metrics['MCC']:.4f}"
        )

