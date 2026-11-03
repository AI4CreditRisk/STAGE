"""
M3 Focal Loss + 标签平滑
"""

import os
import copy
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ============================================================
# 0. 全局配置
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "creditcard"   # 可选：lendingclub / gmsc / internet / creditcard

BATCH_SIZE = 256
EPOCHS = 30

LR = 5e-4
WEIGHT_DECAY = 1e-3

WARMUP_EPOCHS = 0

PATIENCE = 10
MIN_DELTA = 1e-4

CORR_THRESHOLD = 0.6

# Focal Loss 参数
FOCAL_GAMMA = 1.0
LABEL_SMOOTH = 0.05

FILE_MAP = {
    "lendingclub": r"E:\论文写作\creditrisk\论文完整代码\dataset\LoanStatus_2017Q1_ml_01.csv",
    "gmsc": r"E:\论文写作\creditrisk\论文完整代码\dataset\cs_training_fe_01.csv",
    "internet": r"E:\论文写作\creditrisk\论文完整代码\dataset\train_internetdata3.csv",
    "creditcard": r"E:\论文写作\creditrisk\论文完整代码\dataset\default of credit card clientsUCI.csv",
}

print(f"Device  : {DEVICE}")
print(f"Dataset : {DATASET}")
print("Model   : Single-Token Transformer (2 layers, d_model=32, nhead=2)")
print(f"Focal gamma: {FOCAL_GAMMA}, Label Smooth: {LABEL_SMOOTH}")


# ============================================================
# 1. 数据加载与基础清洗（与 baselineeeeee 完全一致）
# ============================================================
def load_and_clean(dataset_name):
    input_file = FILE_MAP[dataset_name]

    # --------------------------------------------------------
    # LendingClub
    # --------------------------------------------------------
    if dataset_name == "lendingclub":
        df = pd.read_csv(input_file, low_memory=False)
        df.columns = df.columns.str.strip()

        target_col = "loan_status"

        default_set = {
            "Charged Off",
            "Default",
            "Late (31-120 days)",
            "In Grace Period",
            "Late (16-30 days)"
        }

        good_set = {
            "Current",
            "Fully Paid"
        }

        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df[df[target_col].isin([0, 1])].copy()
            df[target_col] = df[target_col].astype(int)
        else:
            df[target_col] = df[target_col].astype(str).str.strip()
            valid_status = default_set.union(good_set)
            df = df[df[target_col].isin(valid_status)].copy()
            df[target_col] = df[target_col].apply(lambda x: 1 if x in default_set else 0)

        leakage_cols = [
            "id", "member_id", "url", "desc", "emp_title", "title",
            "funded_amnt", "funded_amnt_inv",
            "out_prncp", "out_prncp_inv",
            "total_pymnt", "total_pymnt_inv",
            "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
            "recoveries", "collection_recovery_fee",
            "last_pymnt_d", "last_pymnt_amnt",
            "next_pymnt_d", "last_credit_pull_d",
            "pymnt_plan", "initial_list_status", "policy_code",
            "hardship_flag", "hardship_type", "hardship_reason",
            "hardship_status", "deferral_term", "hardship_amount",
            "hardship_start_date", "hardship_end_date",
            "payment_plan_start_date", "hardship_length",
            "hardship_dpd", "hardship_loan_status",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
            "debt_settlement_flag",
            "debt_settlement_flag_date",
            "settlement_status",
            "settlement_date",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
            "disbursement_method",
            "revol_bal_joint",
            "sec_app_earliest_cr_line",
            "sec_app_inq_last_6mths",
            "sec_app_mort_acc",
            "sec_app_open_acc",
            "sec_app_revol_util",
            "sec_app_open_act_il",
            "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths",
            "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    # --------------------------------------------------------
    # Give Me Some Credit
    # --------------------------------------------------------
    elif dataset_name == "gmsc":
        df = pd.read_csv(input_file, low_memory=False)
        for col in ["Unnamed: 0", "ID", "id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        target_col = "SeriousDlqin2yrs"
        if target_col not in df.columns:
            alt_cols = [c for c in df.columns if c.lower() == "seriousdlqin2yrs"]
            if len(alt_cols) > 0:
                df.rename(columns={alt_cols[0]: target_col}, inplace=True)
            else:
                raise ValueError(f"未找到目标列：{target_col}")

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    # --------------------------------------------------------
    # Internet Loan Dataset
    # --------------------------------------------------------
    elif dataset_name == "internet":
        df = pd.read_csv(input_file, low_memory=False)
        target_col = "is_default"
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

        leakage_cols = [
            "loan_id",
            "user_id",
            "title",
            "policy_code",
            "early_return",
            "early_return_amount",
            "early_return_amount_3mon"
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    # --------------------------------------------------------
    # UCI Credit Card Default
    # --------------------------------------------------------
    else:
        try:
            df = pd.read_excel(input_file, skiprows=1)
        except Exception:
            df = pd.read_csv(input_file, skiprows=1)

        target_col = "default payment next month"
        if target_col not in df.columns:
            candidates = [c for c in df.columns if "default" in c.lower()]
            if len(candidates) == 0:
                raise ValueError("未找到 default 标签列。")
            target_col = candidates[0]

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    print("\n========== 原始清洗后数据 ==========")
    print(f"数据形状: {df.shape}")
    print(f"标签列: {target_col}")
    print("标签分布:")
    print(df[target_col].value_counts())

    # 基础格式处理（不涉及统计）
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    if "term" in df.columns:
        df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if "interest" in df.columns:
        df["interest"] = pd.to_numeric(df["interest"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    if "recircle_util" in df.columns:
        df["recircle_util"] = pd.to_numeric(df["recircle_util"].astype(str).str.replace("%", "", regex=False), errors="coerce")

    emp_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10
    }
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(emp_map)
    if "work_year" in df.columns:
        df["work_year"] = df["work_year"].map(emp_map)

    target = df[target_col].values.astype(np.int64)
    features = df.drop(columns=[target_col]).copy()
    return features, target


# ============================================================
# 2. 划分数据集
# ============================================================
features_raw, y = load_and_clean(DATASET)

idx_all = np.arange(len(y))
idx_train, idx_temp, y_train_np, y_temp_np = train_test_split(
    idx_all, y, test_size=0.2, random_state=SEED, stratify=y
)
idx_val, idx_test, y_val_np, y_test_np = train_test_split(
    idx_temp, y_temp_np, test_size=0.5, random_state=SEED, stratify=y_temp_np
)

X_train_df = features_raw.iloc[idx_train].copy()
X_val_df = features_raw.iloc[idx_val].copy()
X_test_df = features_raw.iloc[idx_test].copy()

print("\n========== 数据集划分 ==========")
print(f"Train: {X_train_df.shape}")
print(f"Val  : {X_val_df.shape}")
print(f"Test : {X_test_df.shape}")


# ============================================================
# 3. 类别特征编码（仅训练集）
# ============================================================
cat_features = X_train_df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
print(f"\n分类特征数量: {len(cat_features)}")

for col in cat_features:
    train_col = X_train_df[col].fillna("__MISSING__").astype(str)
    train_categories = train_col.unique().tolist()
    category_map = {cat: idx for idx, cat in enumerate(train_categories)}

    X_train_df[col] = train_col.map(category_map).astype(np.int64)
    X_val_df[col] = X_val_df[col].fillna("__MISSING__").astype(str).map(category_map).fillna(-1).astype(np.int64)
    X_test_df[col] = X_test_df[col].fillna("__MISSING__").astype(str).map(category_map).fillna(-1).astype(np.int64)


# ============================================================
# 4. 数值化 + 缺失值用 0 填充
# ============================================================
for col in X_train_df.columns:
    X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
    X_val_df[col] = pd.to_numeric(X_val_df[col], errors="coerce")
    X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce")

X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

# 全部用 0 填补
X_train_df = X_train_df.fillna(0)
X_val_df = X_val_df.fillna(0)
X_test_df = X_test_df.fillna(0)


# ============================================================
# 5. 标准化（仅训练集 fit）
# ============================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns, index=X_train_df.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_df), columns=X_val_df.columns, index=X_val_df.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns, index=X_test_df.index)


# ============================================================
# 6. 相关性过滤（仅训练集）
# ============================================================
corr = X_train_scaled.corr().abs()
drop_set = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if corr.iloc[i, j] > CORR_THRESHOLD:
            drop_set.add(corr.columns[i])

selected_features = [col for col in X_train_scaled.columns if col not in drop_set]
if len(selected_features) == 0:
    selected_features = X_train_scaled.columns.tolist()

X_train_final = X_train_scaled[selected_features].copy()
X_val_final = X_val_scaled[selected_features].copy()
X_test_final = X_test_scaled[selected_features].copy()

print("\n========== 特征筛选结果 ==========")
print(f"原始特征数: {X_train_df.shape[1]}")
print(f"删除高相关特征数: {len(drop_set)}")
print(f"最终特征数: {len(selected_features)}")


# ============================================================
# 7. 转换为 Tensor (batch, 1, feature_num)
# ============================================================
X_train = torch.tensor(X_train_final.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val_final.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test_final.values, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 8. 类别权重（用于 Focal Loss 的 alpha）
# ============================================================
class_weights_np = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_np)
alpha_pos = class_weights_np[1] / (class_weights_np[0] + class_weights_np[1])
alpha_neg = class_weights_np[0] / (class_weights_np[0] + class_weights_np[1])
print(f"\nFocal Alpha: pos={alpha_pos:.4f}, neg={alpha_neg:.4f}")


# ============================================================
# 9. 模型：轻量单 Token Transformer (2层, d_model=32, nhead=2)
# ============================================================
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, nhead=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_enc
        x = self.transformer(x)
        x = x[:, -1, :]          # 取唯一 token
        return self.sigmoid(self.fc(x))


input_dim = X_train.shape[-1]
model = SimpleTransformer(input_dim=input_dim, hidden_dim=32, num_layers=2, nhead=2, dropout=0.3).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {num_params:,}")


# ============================================================
# 10. 损失函数：Focal Loss（加权）+ 标签平滑
# ============================================================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha_pos, alpha_neg, gamma=1.0, smooth=0.05, eps=1e-8):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        # pred: (B,1) 概率, target: (B,1) 0/1
        pred = pred.clamp(self.eps, 1 - self.eps)
        # 标签平滑
        target_smooth = target * (1 - self.smooth) + 0.5 * self.smooth
        # Focal Loss
        pt = pred * target_smooth + (1 - pred) * (1 - target_smooth)
        alpha = self.alpha_pos * target_smooth + self.alpha_neg * (1 - target_smooth)
        loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()


criterion = FocalLossWithSmoothing(alpha_pos, alpha_neg, gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: (epoch+1)/WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1.0
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)


# ============================================================
# 11. 训练（加权 Focal + 平滑 + 梯度裁剪 + 早停）
# ============================================================
best_val_loss = float("inf")
best_state = None
patience_counter = 0

print("\n========== 开始训练 ==========")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)   # (B,1)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)

    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # 验证
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch:02d} | LR: {current_lr:.8f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early Stopping
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"\nRestored best model. Best Val Loss: {best_val_loss:.6f}")


# ============================================================
# 12. 评估部分（增加阈值寻优）
# ============================================================

# ---------- 辅助函数：寻找最佳阈值 ----------
def find_best_threshold(y_true, y_prob, metric='f1', thresholds=None):
    """
    在验证集上搜索使指定指标最优的阈值
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score

# ---------- 辅助函数：计算全部指标 ----------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        'AUC': auc,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Gini': gini,
        'KS': ks,
        'Type II Error': type2,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def print_metrics(m, title):
    print(f"\n================ {title} ================")
    print(f"AUC          : {m['AUC']:.4f}")
    print(f"Accuracy     : {m['Accuracy']:.4f}")
    print(f"Precision    : {m['Precision']:.4f}")
    print(f"Recall       : {m['Recall']:.4f}")
    print(f"F1           : {m['F1']:.4f}")
    print(f"Gini         : {m['Gini']:.4f}")
    print(f"KS           : {m['KS']:.4f}")
    print(f"Type II Error: {m['Type II Error']:.4f}")
    print(f"Confusion Matrix: TN={m['TN']}, FP={m['FP']}, FN={m['FN']}, TP={m['TP']}")

# ---------- 在验证集上寻找最佳阈值 ----------
model.eval()
with torch.no_grad():
    val_probs = model(X_val.to(DEVICE)).cpu().numpy().ravel()
    val_true = y_val.numpy().ravel()
best_thr, best_f1 = find_best_threshold(val_true, val_probs, metric='f1')
print(f"\n验证集最优阈值（F1）: {best_thr:.3f}, 对应F1={best_f1:.4f}")

# ---------- 测试集评估（固定阈值0.5 与 最佳阈值） ----------
with torch.no_grad():
    test_probs = model(X_test.to(DEVICE)).cpu().numpy().ravel()
    test_true = y_test.numpy().ravel()

metrics_05 = compute_metrics(test_true, test_probs, threshold=0.5)
metrics_best = compute_metrics(test_true, test_probs, threshold=best_thr)

print_metrics(metrics_05, "Test @ threshold=0.50")
print_metrics(metrics_best, f"Test @ threshold={best_thr:.3f}")

print("\n最终保留特征数:", len(selected_features))
print("训练集样本数:", len(y_train_np))
print("验证集样本数:", len(y_val_np))
print("测试集样本数:", len(y_test_np))