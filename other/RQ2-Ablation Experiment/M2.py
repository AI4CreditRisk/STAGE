"""Gate Fusion M2
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
    roc_auc_score, precision_score, recall_score,
    accuracy_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ============================================================
# 0. Global configuration
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

DATASET = "creditcard"   # Options: lendingclub / gmsc / internet / creditcard

BATCH_SIZE = 256
EPOCHS = 30

LR = 5e-4
WEIGHT_DECAY = 1e-3

WARMUP_EPOCHS = 5

PATIENCE = 10
MIN_DELTA = 1e-4

CORR_THRESHOLD = 0.6

# Gate entropy regularization coefficient
GATE_ENTROPY_LAMBDA = 0.001

data_dir = r"E:\STAGE\data"

DATASET_CONFIG = {
    'lendingclub': {'path': os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")},
    'gmsc':        {'path': os.path.join(data_dir, "cs_training_fe_01.csv")},
    'internet':    {'path': os.path.join(data_dir, "train_internetdata3.csv")},
    'creditcard':  {'path': os.path.join(data_dir, "default of credit card clientsUCI.csv")},
}

FILE_MAP = {k: v['path'] for k, v in DATASET_CONFIG.items()}


# ============================================================
# 1. Data loading and basic cleaning (raw static features)
# ============================================================
def load_and_clean(dataset_name):
    input_file = FILE_MAP[dataset_name]['path']

    # --------------------------------------------------------
    # LendingClub
    # --------------------------------------------------------
    if dataset_name == "lendingclub":
        df = pd.read_csv(input_file, low_memory=False)
        df.columns = df.columns.str.strip()
        target_col = "loan_status"
        default_set = {"Charged Off", "Default", "Late (31-120 days)",
                       "In Grace Period", "Late (16-30 days)"}
        good_set = {"Current", "Fully Paid"}
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
                raise ValueError(f"Target column not found: {target_col}")
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
            "loan_id", "user_id", "title", "policy_code",
            "early_return", "early_return_amount", "early_return_amount_3mon"
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
                raise ValueError("Default label column not found.")
            target_col = candidates[0]
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    # Basic format handling (no statistics involved)
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
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10
    }
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(emp_map)
    if "work_year" in df.columns:
        df["work_year"] = df["work_year"].map(emp_map)

    target = df[target_col].values.astype(np.int64)
    features = df.drop(columns=[target_col]).copy()
    return features, target


# ============================================================
# 2. Train / validation / test split
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


# ============================================================
# 3. Categorical feature encoding (training set only)
# ============================================================
cat_features = X_train_df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

for col in cat_features:
    train_col = X_train_df[col].fillna("__MISSING__").astype(str)
    train_categories = train_col.unique().tolist()
    category_map = {cat: idx for idx, cat in enumerate(train_categories)}

    X_train_df[col] = train_col.map(category_map).astype(np.int64)
    X_val_df[col] = X_val_df[col].fillna("__MISSING__").astype(str).map(category_map).fillna(-1).astype(np.int64)
    X_test_df[col] = X_test_df[col].fillna("__MISSING__").astype(str).map(category_map).fillna(-1).astype(np.int64)


# ============================================================
# 4. Convert to numeric + fill missing values with 0
# ============================================================
for col in X_train_df.columns:
    X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
    X_val_df[col] = pd.to_numeric(X_val_df[col], errors="coerce")
    X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce")

X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

# Fill all missing values with 0 for train / validation / test
X_train_df = X_train_df.fillna(0)
X_val_df = X_val_df.fillna(0)
X_test_df = X_test_df.fillna(0)


# ============================================================
# 5. Standardization (fit on training set only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns, index=X_train_df.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_df), columns=X_val_df.columns, index=X_val_df.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns, index=X_test_df.index)


# ============================================================
# 6. Correlation filtering (training set only)
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


# ============================================================
# 7. Convert to tensors (batch, 1, feature_num)
# ============================================================
X_train = torch.tensor(X_train_final.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val_final.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test_final.values, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 8. Class weights (for weighted BCE)
# ============================================================
class_weights_np = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_np)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)


# ============================================================
# 9. Model: Transformer + dual projection + gate fusion
# ============================================================
class GateFusionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=2, dropout=0.3):
        super().__init__()
        # Single-token Transformer encoder
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dual projection
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, d_model)

        # Gate network (input: concatenated two projections)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)   # Output two gate weight logits
        )

        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False):
        # x: (batch, 1, input_dim)
        # Transformer encoding
        h = self.embedding(x) + self.pos_enc   # (B, 1, d_model)
        h = self.transformer(h)                # (B, 1, d_model)
        h = h.squeeze(1)                       # (B, d_model)

        # Dual projection
        h1 = self.proj1(h)   # (B, d_model)
        h2 = self.proj2(h)   # (B, d_model)

        # Gating
        gate_logits = self.gate(torch.cat([h1, h2], dim=-1))  # (B, 2)
        gate_w = torch.softmax(gate_logits, dim=-1)           # (B, 2)

        w1 = gate_w[:, 0].unsqueeze(-1)   # (B, 1)
        w2 = gate_w[:, 1].unsqueeze(-1)   # (B, 1)

        h_fused = w1 * h1 + w2 * h2       # (B, d_model)

        y = self.cls(h_fused)             # (B, 1)
        if return_weights:
            return y, gate_w
        return y


# Gate entropy regularization (maximize entropy to encourage uniform gate weights)
def gate_entropy_regularization(gate_w, eps=1e-8):
    ent = -(gate_w * torch.log(gate_w + eps)).sum(dim=-1).mean()
    return -ent   # Negative sign so that a smaller loss means larger entropy


# ============================================================
# 10. Instantiate model
# ============================================================
input_dim = X_train.shape[-1]
model = GateFusionTransformer(input_dim=input_dim, d_model=32, nhead=2,
                              num_layers=2, dropout=0.3).to(DEVICE)


# ============================================================
# 11. Loss, optimizer, scheduler
# ============================================================
criterion = nn.BCELoss(reduction='none')

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1.0
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)


# ============================================================
# 12. Training loop (weighted BCE + gate entropy regularization + gradient clipping + early stopping)
#     Early stopping monitors -val_loss
# ============================================================
best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()

        pred, gate_w = model(xb, return_weights=True)   # pred: (B,1), gate_w: (B,2)

        # Weighted BCE
        sample_weights = class_weights[yb.long().view(-1)].view(-1, 1)
        loss_cls = (criterion(pred, yb) * sample_weights).mean()

        # Gate entropy regularization
        loss_gate = gate_entropy_regularization(gate_w)

        loss = loss_cls + GATE_ENTROPY_LAMBDA * loss_gate

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    # Update learning rate
    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, gate_w = model(xb, return_weights=True)
            sample_weights = class_weights[yb.long().view(-1)].view(-1, 1)
            loss_cls = (criterion(pred, yb) * sample_weights).mean()
            loss_gate = gate_entropy_regularization(gate_w)
            loss = loss_cls + GATE_ENTROPY_LAMBDA * loss_gate
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch:02d} | LR: {current_lr:.8f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early stopping: monitor -val_loss; smaller val_loss is better
    # Use -val_loss as score; larger score is better
    score = -val_loss
    if best_state is None or score > -best_val_loss + MIN_DELTA:
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
# 13. Validation threshold search + test evaluation
# ============================================================
def find_best_threshold(y_true, y_prob, metric='f1', thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


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
        'AUC': auc, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'Gini': gini,
        'KS': ks, 'Type II Error': type2,
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


# Validation threshold search
model.eval()
with torch.no_grad():
    val_probs = []
    val_true = []
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        prob = model(xb).cpu().numpy()
        val_probs.extend(prob)
        val_true.extend(yb.numpy())

val_probs = np.array(val_probs).ravel()
val_true = np.array(val_true).ravel()

best_thr, best_f1 = find_best_threshold(val_true, val_probs, metric='f1')
print(f"\nBest validation threshold (F1): {best_thr:.3f}, F1={best_f1:.4f}")

# Test evaluation
with torch.no_grad():
    test_probs = []
    test_true = []
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        prob = model(xb).cpu().numpy()
        test_probs.extend(prob)
        test_true.extend(yb.numpy())

test_probs = np.array(test_probs).ravel()
test_true = np.array(test_true).ravel()

metrics_05 = compute_metrics(test_true, test_probs, threshold=0.5)
metrics_best = compute_metrics(test_true, test_probs, threshold=best_thr)

print_metrics(metrics_05, "Test @ threshold=0.50")
print_metrics(metrics_best, f"Test @ threshold={best_thr:.3f}")