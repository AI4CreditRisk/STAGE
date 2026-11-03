"""
A Novel Weighted Loss TabTransformer for Imbalanced Credit Risk Datasets
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

# ==================================================================
# Global configuration
# ==================================================================
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'dataset': 'internet',   # 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

    'test_size': 0.1,
    'val_size':  1/9,
    'stratify': True,
    'cat_threshold': 5,

    'norm_range': (-1, 1),
    'fillna_strategy': '0',

    'dim': 128,
    'depth': 1,
    'heads': 2,
    'attn_dropout': 0.1,
    'ff_dropout': 0.1,
    'mlp_hidden_dims': (64, 32),
    'mlp_dropout': 0.3,

    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'num_epochs': 20,
    'monitor': 'val_loss',

    'use_weighted_loss': True,

    'verbose': True,
}


# ==================================================================
# 1. Set random seed for reproducibility
# ==================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['seed'])
DEVICE = torch.device(CONFIG['device'])
print(f"[INFO] Using device: {DEVICE}")


# ==================================================================
# 2. TabTransformer core components
# ==================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=128, heads=2, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q = self.to_q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=2, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ==================================================================
# 3. Main WeightedTabTransformer model
# ==================================================================
class WeightedTabTransformer(nn.Module):
    def __init__(self, categorical_cardinalities, num_continuous,
                 dim=128, depth=1, heads=2, dim_out=2,
                 attn_dropout=0.1, ff_dropout=0.1,
                 mlp_hidden_dims=(64, 32), mlp_dropout=0.3):
        super().__init__()
        self.num_categorical = len(categorical_cardinalities)
        self.num_continuous = num_continuous
        self.dim = dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim) for card in categorical_cardinalities
        ])

        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])

        mlp_input_dim = self.num_categorical * dim + num_continuous
        h1, h2 = mlp_hidden_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, h1),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(h2, dim_out),
        )

    def forward(self, cat_x, num_x):
        if self.num_categorical > 0:
            embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.stack(embeds, dim=1)
            for block in self.transformer:
                x = block(x)
            x = x.reshape(x.size(0), -1)
            if self.num_continuous > 0:
                x = torch.cat([x, num_x], dim=1)
        else:
            x = num_x
        return self.mlp(x)


# ==================================================================
# 4. Dataset class
# ==================================================================
class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, labels):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_data[idx], self.num_data[idx], self.labels[idx]


# ==================================================================
# 5. Data preprocessing
# ==================================================================
def prepare_data(X_train, X_val, X_test, y_train, y_val, y_test,
                 cat_cols, num_cols, batch_size=64,
                 norm_range=(-1, 1), fillna_strategy='0'):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    cardinalities = []
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna('missing')
        X_val[col] = X_val[col].astype(str).fillna('missing')
        X_test[col] = X_test[col].astype(str).fillna('missing')

        le = LabelEncoder()
        le.fit(X_train[col])
        known = set(le.classes_)

        n_classes = len(le.classes_)
        X_train[col] = le.transform(X_train[col])
        X_val[col] = X_val[col].apply(lambda x: le.transform([x])[0] if x in known else n_classes)
        X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in known else n_classes)

        cardinalities.append(n_classes + 1)

    if len(num_cols) > 0:
        for col in num_cols:
            if fillna_strategy == 'median':
                fill_val = X_train[col].median()
            elif fillna_strategy == 'mean':
                fill_val = X_train[col].mean()
            else:
                fill_val = 0
            X_train[col] = X_train[col].fillna(fill_val)
            X_val[col] = X_val[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

        scaler = MinMaxScaler(feature_range=norm_range)
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    cat_train = X_train[cat_cols].values.astype(np.int64) if len(cat_cols) > 0 else np.zeros((len(X_train), 0), dtype=np.int64)
    cat_val = X_val[cat_cols].values.astype(np.int64) if len(cat_cols) > 0 else np.zeros((len(X_val), 0), dtype=np.int64)
    cat_test = X_test[cat_cols].values.astype(np.int64) if len(cat_cols) > 0 else np.zeros((len(X_test), 0), dtype=np.int64)

    num_train = X_train[num_cols].values.astype(np.float32) if len(num_cols) > 0 else np.zeros((len(X_train), 0), dtype=np.float32)
    num_val = X_val[num_cols].values.astype(np.float32) if len(num_cols) > 0 else np.zeros((len(X_val), 0), dtype=np.float32)
    num_test = X_test[num_cols].values.astype(np.float32) if len(num_cols) > 0 else np.zeros((len(X_test), 0), dtype=np.float32)

    y_train_arr = np.array(y_train).astype(np.int64)
    y_val_arr = np.array(y_val).astype(np.int64)
    y_test_arr = np.array(y_test).astype(np.int64)

    train_ds = TabularDataset(cat_train, num_train, y_train_arr)
    val_ds = TabularDataset(cat_val, num_val, y_val_arr)
    test_ds = TabularDataset(cat_test, num_test, y_test_arr)

    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, cardinalities, y_train_arr


# ==================================================================
# 6. Class weight computation
# ==================================================================
def compute_class_weights(y_train, num_classes):
    N = len(y_train)
    weights = []
    for i in range(num_classes):
        N_i = np.sum(y_train == i)
        w_i = N / N_i if N_i > 0 else 1.0
        weights.append(w_i)
    weights = np.array(weights, dtype=np.float32)
    print(f"[INFO] Class distribution: {np.bincount(y_train)}")
    print(f"[INFO] Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


# ==================================================================
# 7. Training function (no early stopping)
# ==================================================================
def train_model(model, train_loader, val_loader, class_weights,
                num_epochs=100, lr=1e-3, weight_decay=0.0, verbose=True):
    model = model.to(DEVICE)

    if CONFIG['use_weighted_loss']:
        class_weights = class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        model.train()
        train_loss_total, n_train = 0.0, 0

        for cat_x, num_x, y in train_loader:
            cat_x, num_x, y = cat_x.to(DEVICE), num_x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(cat_x, num_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * y.size(0)
            n_train += y.size(0)

        train_loss = train_loss_total / n_train

        model.eval()
        val_loss_total, n_val = 0.0, 0
        with torch.no_grad():
            for cat_x, num_x, y in val_loader:
                cat_x, num_x, y = cat_x.to(DEVICE), num_x.to(DEVICE), y.to(DEVICE)
                logits = model(cat_x, num_x)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * y.size(0)
                n_val += y.size(0)
        val_loss = val_loss_total / n_val

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        if verbose:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.4f}s")

    avg_epoch_time = np.mean(epoch_times)
    return model, avg_epoch_time


# ==================================================================
# 8. Evaluation function
# ==================================================================
def evaluate_model(model, test_loader, avg_epoch_time):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for cat_x, num_x, y in test_loader:
            cat_x, num_x = cat_x.to(DEVICE), num_x.to(DEVICE)
            logits = model(cat_x, num_x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    pos_probs = all_probs[:, 1]
    auc = roc_auc_score(all_labels, pos_probs)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(all_labels, pos_probs)
    ks = np.max(tpr - fpr)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("Model Evaluation Results (Test Set)")
    print("=" * 60)
    print(f"AUC:                     {auc:.4f}")
    print(f"Accuracy:                {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:               {prec:.4f}")
    print(f"Recall:                  {rec:.4f}")
    print(f"F1:                      {f1:.4f}")
    print(f"Gini:                    {gini:.4f}")
    print(f"KS:                      {ks:.4f}")
    print(f"Avg Training Time/Epoch: {avg_epoch_time:.4f} s")
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    return {
        'AUC': auc, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'Gini': gini, 'KS': ks,
        'AvgEpochTime': avg_epoch_time
    }


# ==================================================================
# 9. Automatic feature type inference
# ==================================================================
def infer_feature_types(X, cat_threshold=5):
    cat_cols = []
    num_cols = []
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(X[col]) and X[col].nunique(dropna=True) <= cat_threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    print(f"[INFO] Number of categorical features: {len(cat_cols)}")
    print(f"[INFO] Number of numerical features:   {len(num_cols)}")
    return cat_cols, num_cols


# ==================================================================
# Load data
# ==================================================================
DATASET = CONFIG['dataset']   # 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

# ---------------- Dataset 1: lendingclub ----------------
if DATASET == 'lendingclub':
    input_csv = os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    default_set = {'Charged Off', 'Default', 'Late (31-120 days)',
                   'In Grace Period', 'Late (16-30 days)'}
    good_set = {'Current', 'Fully Paid'}

    if pd.api.types.is_numeric_dtype(df['loan_status']):
        df['loan_status'] = pd.to_numeric(df['loan_status'], errors='coerce')
        df = df[df['loan_status'].isin([0, 1])].copy()
        df['loan_status'] = df['loan_status'].astype(int)
    else:
        df['loan_status'] = df['loan_status'].astype(str).str.strip()
        valid_status = default_set.union(good_set)
        df = df[df['loan_status'].isin(valid_status)].copy()
        df['loan_status'] = df['loan_status'].apply(
            lambda x: 1 if x in default_set else 0)

    leakage_cols = [
        'id', 'member_id', 'url', 'desc', 'emp_title', 'title',
        'funded_amnt', 'funded_amnt_inv',
        'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv',
        'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt',
        'next_pymnt_d', 'last_credit_pull_d',
        'pymnt_plan', 'initial_list_status', 'policy_code',
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date',
        'hardship_end_date', 'payment_plan_start_date', 'hardship_length',
        'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
        'debt_settlement_flag', 'debt_settlement_flag_date',
        'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term',
        'disbursement_method',
        'revol_bal_joint', 'sec_app_earliest_cr_line',
        'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
        'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
        'sec_app_chargeoff_within_12_mths',
        'sec_app_collections_12_mths_ex_med',
        'sec_app_mths_since_last_major_derog',
    ]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Deleted {len(drop_cols)} leakage columns, remaining {df.shape[1]} columns")

    if 'int_rate' in df.columns:
        df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'revol_util' in df.columns:
        df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    if 'emp_length' in df.columns:
        emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                   '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                   '8 years': 8, '9 years': 9, '10+ years': 10}
        df['emp_length'] = df['emp_length'].map(emp_map)

    issue_d = pd.to_datetime(df.get('issue_d'), format='%b-%Y', errors='coerce')
    earliest_cr = pd.to_datetime(df.get('earliest_cr_line'), format='%b-%Y', errors='coerce')
    if issue_d is not None and earliest_cr is not None:
        df['credit_history_months'] = ((issue_d - earliest_cr).dt.days / 30.44).fillna(0).clip(lower=0)
    else:
        df['credit_history_months'] = 0.0

    inq_12 = df.get('inq_last_12m', pd.Series(0, index=df.index)).fillna(0)
    inq_6 = df.get('inq_last_6mths', pd.Series(0, index=df.index)).fillna(0)
    df['inq_acceleration'] = np.where(inq_12 > 0, inq_6 / (inq_12 + 1e-8), 0)

    open_24 = df.get('acc_open_past_24mths', pd.Series(0, index=df.index)).fillna(0)
    open_6 = df.get('open_acc_6m', pd.Series(0, index=df.index)).fillna(0)
    df['open_acceleration'] = np.where(open_24 > 0, open_6 / (open_24 + 1e-8), 0)

    for c in ['issue_d', 'earliest_cr_line']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    target_col = 'loan_status'

# ---------------- Dataset 2: gmsc ----------------
elif DATASET == 'gmsc':
    input_csv = os.path.join(data_dir, "cs_training_fe_01.csv")
    df = pd.read_csv(input_csv)
    target_col = 'SeriousDlqin2yrs'

# ---------------- Dataset 3: internet ----------------
elif DATASET == 'internet':
    input_csv = os.path.join(data_dir, "train_internetdata3.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Original data shape: {df.shape}")

    target_col = 'is_default'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    print(f"Label distribution:\n{df[target_col].value_counts()}")

    leakage_cols = [
        'loan_id', 'user_id', 'title', 'policy_code',
        'early_return', 'early_return_amount', 'early_return_amount_3mon',
    ]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Deleted {len(drop_cols)} leakage/useless columns, remaining {df.shape[1]} columns")

# ---------------- Dataset 4: creditcard ----------------
elif DATASET == 'creditcard':
    input_csv = os.path.join(data_dir, "default of credit card clientsUCI.csv")
    try:
        df = pd.read_excel(input_csv, skiprows=1)
    except Exception:
        df = pd.read_csv(input_csv, skiprows=1)
    print("Original data shape:", df.shape)

    target_col = 'default payment next month'
    if target_col not in df.columns:
        cand = [c for c in df.columns if 'default' in c.lower()]
        target_col = cand[0]
        print("Using label column:", target_col)

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

else:
    raise ValueError("Unknown dataset name")


# ==================================================================
# Feature/label split + partitioning + training + evaluation
# ==================================================================
# Drop columns that are entirely NaN
df = df.dropna(axis=1, how='all')

y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# Attempt to convert object columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='ignore')

print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
print(f"[INFO] Target distribution:\n{y.value_counts()}")

# Infer categorical and numerical features
cat_cols, num_cols = infer_feature_types(X, cat_threshold=CONFIG['cat_threshold'])

# 8:1:1 split
stratify_y = y if CONFIG['stratify'] else None
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=CONFIG['test_size'],
    random_state=CONFIG['seed'], stratify=stratify_y
)
stratify_tv = y_train_val if CONFIG['stratify'] else None
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=CONFIG['val_size'],
    random_state=CONFIG['seed'], stratify=stratify_tv
)

print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Preprocessing
train_loader, val_loader, test_loader, cardinalities, y_train_arr = prepare_data(
    X_train, X_val, X_test, y_train, y_val, y_test,
    cat_cols=cat_cols, num_cols=num_cols,
    batch_size=CONFIG['batch_size'],
    norm_range=CONFIG['norm_range'],
    fillna_strategy=CONFIG['fillna_strategy'],
)

# Class weights
class_weights = compute_class_weights(y_train_arr, num_classes=2)

# Build model
model = WeightedTabTransformer(
    categorical_cardinalities=cardinalities,
    num_continuous=len(num_cols),
    dim=CONFIG['dim'],
    depth=CONFIG['depth'],
    heads=CONFIG['heads'],
    dim_out=2,
    attn_dropout=CONFIG['attn_dropout'],
    ff_dropout=CONFIG['ff_dropout'],
    mlp_hidden_dims=CONFIG['mlp_hidden_dims'],
    mlp_dropout=CONFIG['mlp_dropout'],
)
print(model)

# Train (no early stopping, runs for fixed num_epochs)
model, avg_epoch_time = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    class_weights=class_weights,
    num_epochs=CONFIG['num_epochs'],
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    verbose=CONFIG['verbose'],
)

# Test evaluation
results = evaluate_model(model, test_loader, avg_epoch_time)