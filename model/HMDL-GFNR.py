# -*- coding: utf-8 -*-
"""
HMDL-GFNR: A Hybrid Multi-Stage Deep Learning Model with
Graph-Based Financial Network Representation for Credit Risk Assessment
"""

import os
import re
import time
import random
import warnings
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

# ==================================================================
# 0. Fixed random seed
# ==================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ==================================================================
# Dataset selection
# Options: 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'
# ==================================================================
DATASET = "lendingclub"
# ==================================================================
# 1. Data loading
# ==================================================================
def load_custom_data(dataset_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Load and preprocess the selected dataset.

    Returns:
        df: Processed DataFrame
        target_col: Name of the target column
    """
    dataset_name = dataset_name.lower().strip()

    # ---------------- Dataset 1: lendingclub ----------------
    if dataset_name == "lendingclub":
        input_csv = os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")
        df = pd.read_csv(input_csv, low_memory=False)
        df.columns = df.columns.str.strip()
        default_set = {
            "Charged Off", "Default", "Late (31-120 days)",
            "In Grace Period", "Late (16-30 days)"
        }
        good_set = {"Current", "Fully Paid"}

        if pd.api.types.is_numeric_dtype(df["loan_status"]):
            df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
            df = df[df["loan_status"].isin([0, 1])].copy()
            df["loan_status"] = df["loan_status"].astype(int)
        else:
            df["loan_status"] = df["loan_status"].astype(str).str.strip()
            valid_status = default_set.union(good_set)
            df = df[df["loan_status"].isin(valid_status)].copy()
            df["loan_status"] = df["loan_status"].apply(lambda x: 1 if x in default_set else 0)

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
            "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
            "deferral_term", "hardship_amount", "hardship_start_date",
            "hardship_end_date", "payment_plan_start_date", "hardship_length",
            "hardship_dpd", "hardship_loan_status",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount", "hardship_last_payment_amount",
            "debt_settlement_flag", "debt_settlement_flag_date",
            "settlement_status", "settlement_date",
            "settlement_amount", "settlement_percentage", "settlement_term",
            "disbursement_method",
            "revol_bal_joint", "sec_app_earliest_cr_line",
            "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
            "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths",
            "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        print(f"[LendingClub] Dropped {len(drop_cols)} leakage columns, remaining {df.shape[1]} columns")

        if "int_rate" in df.columns:
            df["int_rate"] = pd.to_numeric(
                df["int_rate"].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )
        if "revol_util" in df.columns:
            df["revol_util"] = pd.to_numeric(
                df["revol_util"].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )
        if "term" in df.columns:
            df["term"] = pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce"
            )
        if "emp_length" in df.columns:
            emp_map = {
                "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
                "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                "8 years": 8, "9 years": 9, "10+ years": 10
            }
            df["emp_length"] = df["emp_length"].map(emp_map)

        issue_d = pd.to_datetime(df.get("issue_d"), format="%b-%Y", errors="coerce")
        earliest_cr = pd.to_datetime(df.get("earliest_cr_line"), format="%b-%Y", errors="coerce")
        if issue_d is not None and earliest_cr is not None:
            df["credit_history_months"] = ((issue_d - earliest_cr).dt.days / 30.44).fillna(0).clip(lower=0)
        else:
            df["credit_history_months"] = 0.0

        inq_12 = df.get("inq_last_12m", pd.Series(0, index=df.index)).fillna(0)
        inq_6 = df.get("inq_last_6mths", pd.Series(0, index=df.index)).fillna(0)
        df["inq_acceleration"] = np.where(inq_12 > 0, inq_6 / (inq_12 + 1e-8), 0)

        open_24 = df.get("acc_open_past_24mths", pd.Series(0, index=df.index)).fillna(0)
        open_6 = df.get("open_acc_6m", pd.Series(0, index=df.index)).fillna(0)
        df["open_acceleration"] = np.where(open_24 > 0, open_6 / (open_24 + 1e-8), 0)

        for c in ["issue_d", "earliest_cr_line"]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        target_col = "loan_status"
        return df, target_col

    # ---------------- Dataset 2: gmsc ----------------
    elif dataset_name == "gmsc":
        input_csv = os.path.join(data_dir, "cs_training_fe_01.csv")
        df = pd.read_csv(input_csv)
        target_col = "SeriousDlqin2yrs"
        return df, target_col

    # ---------------- Dataset 3: internet ----------------
    elif dataset_name == "internet":
        input_csv = os.path.join(data_dir, "train_internetdata3.csv")
        df = pd.read_csv(input_csv, low_memory=False)
        print(f"[Internet] Raw data shape: {df.shape}")

        target_col = "is_default"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)
        print(f"[Internet] Label distribution:\n{df[target_col].value_counts()}")

        leakage_cols = [
            "loan_id", "user_id", "title", "policy_code",
            "early_return", "early_return_amount", "early_return_amount_3mon",
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        print(f"[Internet] Dropped {len(drop_cols)} leakage/unused columns, remaining {df.shape[1]} columns")

        return df, target_col

    # ---------------- Dataset 4: creditcard ----------------
    elif dataset_name == "creditcard":
        input_csv = os.path.join(data_dir, "default of credit card clientsUCI.csv")
        try:
            df = pd.read_excel(input_csv, skiprows=1)
        except Exception:
            df = pd.read_csv(input_csv, skiprows=1)
        print(f"[CreditCard] Raw data shape: {df.shape}")

        target_col = "default payment next month"
        if target_col not in df.columns:
            cand = [c for c in df.columns if "default" in c.lower()]
            if len(cand) == 0:
                raise ValueError("creditcard dataset does not contain a default label column")
            target_col = cand[0]
            print(f"[CreditCard] Using label column: {target_col}")

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

        return df, target_col

    else:
        raise ValueError("DATASET must be one of 'lendingclub', 'gmsc', 'internet', 'creditcard'")


# ==================================================================
# 2. General preprocessing
# ==================================================================
def preprocess_dataframe(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    General preprocessing:
    1. Separate labels
    2. Encode object/category columns with LabelEncoder
    3. Convert numeric columns to float
    4. Fill missing values with median
    5. Normalize with MinMaxScaler
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")

    y = df[target_col].astype(int).values
    X_df = df.drop(columns=[target_col]).copy()

    # Remove duplicate column names if any
    X_df = X_df.loc[:, ~X_df.columns.duplicated()]

    # Convert bool to int
    for c in X_df.columns:
        if X_df[c].dtype == bool:
            X_df[c] = X_df[c].astype(int)

    # Encode object/category columns
    for c in X_df.columns:
        if str(X_df[c].dtype) in ["object", "category"]:
            X_df[c] = X_df[c].astype(str).fillna("NA")
            le = LabelEncoder()
            X_df[c] = le.fit_transform(X_df[c])

    # Force numeric
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # Fill missing values
    for c in X_df.columns:
        if X_df[c].isna().all():
            X_df[c] = 0.0
        else:
            median_val = X_df[c].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_df[c] = X_df[c].fillna(median_val)

    X_df = X_df.replace([np.inf, -np.inf], 0)

    # Keep constant columns; no extra deletion to preserve feature space
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df.values.astype(np.float32))

    return X.astype(np.float32), y.astype(np.int64)


# ==================================================================
# 3. Dataset class
# ==================================================================
class CreditRiskDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================================================================
# 4. Pseudo-sequence builder for LSTM and Transformer
# ==================================================================
class PseudoSequenceBuilder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.step_dim = int(np.ceil(input_dim / seq_len))
        self.total_dim = self.seq_len * self.step_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        return: [B, T, step_dim]
        """
        batch_size, feat_dim = x.shape
        if feat_dim < self.total_dim:
            pad_dim = self.total_dim - feat_dim
            x = F.pad(x, (0, pad_dim), mode="constant", value=0.0)
        elif feat_dim > self.total_dim:
            x = x[:, :self.total_dim]
        x = x.view(batch_size, self.seq_len, self.step_dim)
        return x


# ==================================================================
# 5. GNN / GCN modules (Graph-Based Borrower Network Representation)
# ==================================================================
class GCNLayer(nn.Module):
    """
    Standard GCN layer: H' = ReLU(A_hat H W + b)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = torch.matmul(a_hat, h)
        return F.relu(h)


class GraphEncoder(nn.Module):
    """
    Builds a kNN graph within each mini-batch to simulate a borrower-lender financial network.
    This is the closest practical implementation for tabular data without explicit relational edges.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64, k_neighbors: int = 8):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def build_normalized_adj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a symmetric kNN adjacency matrix using cosine similarity,
        then add self-loops and normalize: A -> A + I -> D^{-1/2} A D^{-1/2}
        """
        bsz = x.size(0)

        x_norm = F.normalize(x, p=2, dim=1)
        sim = torch.matmul(x_norm, x_norm.t())  # [B, B]

        k = min(self.k_neighbors, bsz)
        _, topk_idx = torch.topk(sim, k=k, dim=1)

        A = torch.zeros((bsz, bsz), device=x.device, dtype=torch.float32)
        A.scatter_(1, topk_idx, 1.0)

        # Symmetrize
        A = torch.maximum(A, A.t())

        # Add self-loops
        I = torch.eye(bsz, device=x.device)
        A = A + I

        degree = torch.sum(A, dim=1)
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
        return A_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_hat = self.build_normalized_adj(x)
        h1 = self.gcn1(x, a_hat)
        h2 = self.gcn2(h1, a_hat)
        return h2


# ==================================================================
# 6. Transformer Temporal Encoder
# Corresponds to equations (7), (8), (9) in the paper:
#   (7) Q = XW_Q, K = XW_K, V = XW_V
#   (9) alpha_t = exp(e_t) / sum_k exp(e_k)
#   (8) H_transformer = sum_t alpha_t * X_t
# ==================================================================
class TransformerTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 8, d_model: int = 64):
        super().__init__()
        self.sequence_builder = PseudoSequenceBuilder(input_dim=input_dim, seq_len=seq_len)
        self.seq_len = seq_len
        self.step_dim = self.sequence_builder.step_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(self.step_dim, d_model, bias=True)

        # Trainable weight matrices for Q, K, V (Eq. 7)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X_seq: [B, T, step_dim]
        x_seq = self.sequence_builder(x)

        # Project to d_model
        x_seq = self.input_proj(x_seq)  # [B, T, d_model]

        # Eq. (7)
        Q = self.W_Q(x_seq)  # [B, T, d_model]
        K = self.W_K(x_seq)  # [B, T, d_model]
        V = self.W_V(x_seq)  # [B, T, d_model]

        # Scaled dot-product attention scores
        d_k = self.d_model
        scores = torch.matmul(Q, K.transpose(1, 2)) / (d_k ** 0.5)  # [B, T, T]

        # Compute energy e_t for each time step by averaging over keys
        e_t = scores.mean(dim=-1)  # [B, T]

        # Eq. (9): alpha_t
        alpha_t = torch.softmax(e_t, dim=-1)  # [B, T]

        # Contextualized sequence
        context_seq = torch.matmul(torch.softmax(scores, dim=-1), V)  # [B, T, d_model]

        # Eq. (8): H_transformer = sum_t alpha_t * X_t
        H_transformer = torch.sum(alpha_t.unsqueeze(-1) * context_seq, dim=1)  # [B, d_model]

        return H_transformer


# ==================================================================
# 7. LSTM Temporal Encoder
# ==================================================================
class LSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 8, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.sequence_builder = PseudoSequenceBuilder(input_dim=input_dim, seq_len=seq_len)
        self.step_dim = self.sequence_builder.step_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = self.sequence_builder(x)  # [B, T, step_dim]
        output, (h_n, c_n) = self.lstm(x_seq)
        H_lstm = output[:, -1, :]  # Last time step representation
        return H_lstm


# ==================================================================
# 8. CNN Encoder
# ==================================================================
class CNNEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveMaxPool1d(output_size=4)
        self.fc = nn.Linear(64 * 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)  # [B, 1, D]
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.pool(h)
        h = h.flatten(start_dim=1)
        H_cnn = F.relu(self.fc(h))
        return H_cnn


# ==================================================================
# 9. Feature fusion layer
# ==================================================================
class FeatureFusionLayer(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.W_g = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_l = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_c = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_t = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(
        self,
        H_gcn: torch.Tensor,
        H_lstm: torch.Tensor,
        H_cnn: torch.Tensor,
        H_transformer: torch.Tensor
    ) -> torch.Tensor:
        fused = (
            self.W_g(H_gcn) +
            self.W_l(H_lstm) +
            self.W_c(H_cnn) +
            self.W_t(H_transformer)
        )
        embed_features = torch.sigmoid(fused)
        return embed_features


# ==================================================================
# 10. HMDL-GFNR main model
# ==================================================================
class HMDLGFNR(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 64, seq_len: int = 8, num_classes: int = 2):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=emb_dim,
            output_dim=emb_dim,
            k_neighbors=8
        )

        self.transformer_encoder = TransformerTemporalEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=emb_dim
        )

        self.lstm_encoder = LSTMTemporalEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=emb_dim,
            num_layers=1
        )

        self.cnn_encoder = CNNEncoder(
            input_dim=input_dim,
            output_dim=emb_dim
        )

        self.fusion_layer = FeatureFusionLayer(emb_dim=emb_dim)

        # Auxiliary classification head for PyTorch supervision
        self.classifier_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim // 2, num_classes)
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        H_gcn = self.graph_encoder(x)
        H_lstm = self.lstm_encoder(x)
        H_cnn = self.cnn_encoder(x)
        H_transformer = self.transformer_encoder(x)

        embed_features = self.fusion_layer(
            H_gcn=H_gcn,
            H_lstm=H_lstm,
            H_cnn=H_cnn,
            H_transformer=H_transformer
        )
        return embed_features

    def forward(self, x: torch.Tensor):
        embed_features = self.extract_embedding(x)
        logits = self.classifier_head(embed_features)
        return logits, embed_features


# ==================================================================
# 11. Metric computation
# ==================================================================
def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS = max(TPR - FPR), commonly used in credit scoring.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return float(ks)


def compute_type2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Type II Error / FNR = FN / (FN + TP)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(fn / (fn + tp + 1e-12))


def evaluate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    gini = 2 * auc - 1
    ks = compute_ks_statistic(y_true, y_prob)
    type2 = compute_type2_error(y_true, y_pred)

    metrics = {
        "AUC": auc,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Gini Coefficient": gini,
        "KS Statistic": ks,
        "Type II Error (FNR)": type2
    }
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for k, v in metrics.items():
        print(f"{k:<25s}: {v:.6f}")


# ==================================================================
# 12. Training function for feature extractor
# ==================================================================
def train_feature_extractor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 20,
    lr: float = 1e-3
):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()

        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, embed_features = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        avg_loss = total_loss / max(total_samples, 1)
        print(f"[Epoch {epoch:03d}/{epochs:03d}] Loss: {avg_loss:.6f} | Time: {epoch_time:.4f} seconds")

        if val_loader is not None:
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(DEVICE)
                    logits, _ = model(batch_x)
                    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    val_probs.append(prob)
                    val_labels.append(batch_y.numpy())
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            val_auc = roc_auc_score(val_labels, val_probs)
            print(f"                 Val AUC: {val_auc:.6f}")

    avg_epoch_time = float(np.mean(epoch_times))
    print(f"\nAverage Training Time per Epoch: {avg_epoch_time:.4f} seconds")
    return avg_epoch_time


# ==================================================================
# 13. Embedding extraction
# ==================================================================
def extract_embeddings(model: nn.Module, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            embed_features = model.extract_embedding(batch_x)
            all_embeddings.append(embed_features.detach().cpu().numpy())
            all_labels.append(batch_y.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


# ==================================================================
# 14. CatBoost final classifier (Section 4.7 hyperparameters)
# ==================================================================
def train_catboost_classifier(X_train_emb: np.ndarray, y_train: np.ndarray) -> CatBoostClassifier:
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.2,
        depth=10,
        l2_leaf_reg=3.0,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        random_seed=SEED,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False
    )
    cat_model.fit(X_train_emb, y_train)
    return cat_model


# ==================================================================
# 15. Main pipeline
# ==================================================================
def main():
    print(f"[INFO] Selected dataset: {DATASET}")

    # ---------------- 1) Load data ----------------
    df, target_col = load_custom_data(DATASET)
    print(f"[INFO] Loaded Data Shape: {df.shape}")
    print(f"[INFO] Target Column: {target_col}")

    # ---------------- 2) Preprocess ----------------
    X, y = preprocess_dataframe(df, target_col)
    print(f"[INFO] Feature Matrix Shape: {X.shape}")
    print(f"[INFO] Label Shape: {y.shape}")
    print(f"[INFO] Positive Ratio: {np.mean(y):.6f}")

    # ---------------- 3) Split data ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train
    )

    train_dataset = CreditRiskDataset(X_train_sub, y_train_sub)
    val_dataset = CreditRiskDataset(X_val, y_val)
    full_train_dataset = CreditRiskDataset(X_train, y_train)
    test_dataset = CreditRiskDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
    full_train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    # ---------------- 4) Build HMDL-GFNR ----------------
    input_dim = X.shape[1]
    model = HMDLGFNR(
        input_dim=input_dim,
        emb_dim=64,
        seq_len=8,
        num_classes=2
    ).to(DEVICE)

    print(model)

    # ---------------- 5) Train deep feature extractor ----------------
    _ = train_feature_extractor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=1e-3
    )

    # ---------------- 6) Extract embeddings ----------------
    X_train_emb, y_train_emb = extract_embeddings(model, full_train_loader)
    X_test_emb, y_test_emb = extract_embeddings(model, test_loader)

    print(f"[INFO] Train Embedding Shape: {X_train_emb.shape}")
    print(f"[INFO] Test Embedding Shape: {X_test_emb.shape}")

    # ---------------- 7) CatBoost final classifier ----------------
    cat_model = train_catboost_classifier(X_train_emb, y_train_emb)

    # ---------------- 8) Evaluate on test set ----------------
    y_test_prob = cat_model.predict_proba(X_test_emb)[:, 1]
    metrics = evaluate_all_metrics(y_test_emb, y_test_prob, threshold=0.5)
    print_metrics(metrics, title="HMDL-GFNR Final Test Metrics")

    # Additional confusion matrix
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test_emb, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()