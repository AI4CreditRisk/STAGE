# Enhanced Bank Lending Decisions: A Deep Learning Strategy with Feature Selection Based on Lightweight Parrot Bi-LSTM
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import ks_2samp
from scipy.special import gamma as gamma_func
import warnings
import time
import atexit
from contextlib import contextmanager
import math
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==============================
# Global script runtime statistics
# ==============================
_GLOBAL_START_TIME = time.perf_counter()

def _format_seconds(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.2f}s"
    elif m > 0:
        return f"{m}m {s:.2f}s"
    else:
        return f"{s:.2f}s"

def print_total_runtime():
    total_time = time.perf_counter() - _GLOBAL_START_TIME
    print("=" * 60)
    print(f"Total Runtime: {_format_seconds(total_time)}")
    print("=" * 60)

atexit.register(print_total_runtime)


@contextmanager
def timer(name="Code Block", use_cuda_sync=True):
    try:
        if use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    start = time.perf_counter()
    print(f"[Timer Start] {name}")
    try:
        yield
    finally:
        try:
            if use_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        elapsed = time.perf_counter() - start
        print(f"[Timer End] {name}: {_format_seconds(elapsed)}")


# ==============================
# Dataset configuration
# ==============================
DATASET = 'internet'

DATASET_CONFIG = {
    'lendingclub': {'path': os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")},
    'gmsc':        {'path': os.path.join(data_dir, "cs_training_fe_01.csv")},
    'internet':    {'path': os.path.join(data_dir, "train_internetdata3.csv")},
    'creditcard':  {'path': os.path.join(data_dir, "default of credit card clientsUCI.csv")},
}


# ============================================================
# 1. INPUT: Data loading and cleaning
# ============================================================
def load_and_clean(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    path = cfg['path']
    print(f"\n{'='*60}\n[INPUT] Dataset: {dataset_name} | File: {path}\n{'='*60}")

    if dataset_name == 'creditcard':
        try:
            df = pd.read_excel(path, skiprows=1)
        except Exception:
            df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path, low_memory=False)
    print(f"Raw data shape: {df.shape}")

    if dataset_name == 'lendingclub':
        target_col = 'loan_status'
        default_set = {'Charged Off', 'Default', 'Late (31-120 days)',
                       'In Grace Period', 'Late (16-30 days)'}
        good_set = {'Current', 'Fully Paid'}
        df = df[df[target_col].isin(default_set.union(good_set))].copy()
        df[target_col] = df[target_col].apply(lambda x: 1 if x in default_set else 0)

    elif dataset_name == 'gmsc':
        candidates = ['SeriousDlqin2yrs', 'seriousdlqin2yrs', 'target', 'label', 'default']
        target_col = next((c for c in candidates if c in df.columns), None)
        if target_col is None:
            raise ValueError(f"Target column not found. Columns: {df.columns.tolist()}")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].notna()].copy()
        df[target_col] = df[target_col].astype(int)
        df = df[df[target_col].isin([0, 1])].copy()

    elif dataset_name == 'internet':
        target_col = 'is_default'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

    elif dataset_name == 'creditcard':
        target_col = 'default payment next month'
        if target_col not in df.columns:
            cand = [c for c in df.columns if 'default' in c.lower()]
            target_col = cand[0]
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

    print(f"Identified target column: {target_col}")
    print(f"Label distribution:\n{df[target_col].value_counts()}")

    if dataset_name in ('lendingclub', 'gmsc'):
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
    elif dataset_name == 'internet':
        leakage_cols = [
            'loan_id', 'user_id', 'title', 'policy_code',
            'early_return', 'early_return_amount', 'early_return_amount_3mon',
        ]
    else:
        leakage_cols = []

    leakage_exist = [c for c in leakage_cols if c in df.columns]
    if leakage_exist:
        df = df.drop(columns=leakage_exist)
    print(f"Dropped {len(leakage_exist)} leakage columns, remaining {df.shape[1]} columns")

    for col in ['int_rate', 'revol_util']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(
            df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    if 'interest' in df.columns:
        df['interest'] = pd.to_numeric(
            df['interest'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'recircle_util' in df.columns:
        df['recircle_util'] = pd.to_numeric(
            df['recircle_util'].astype(str).str.replace('%', '', regex=False), errors='coerce')

    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
               '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
               '8 years': 8, '9 years': 9, '10+ years': 10}
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].map(emp_map)
    if 'work_year' in df.columns:
        df['work_year'] = df['work_year'].map(emp_map)

    if dataset_name in ('lendingclub', 'gmsc'):
        if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
            issue_d = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            earliest_cr = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            df['credit_history_months'] = ((issue_d - earliest_cr).dt.days / 30.44
                                            ).fillna(0).clip(lower=0)
        else:
            df['credit_history_months'] = 0

        inq_12 = df.get('inq_last_12m', pd.Series(0, index=df.index)).fillna(0)
        inq_6 = df.get('inq_last_6mths', pd.Series(0, index=df.index)).fillna(0)
        df['inq_acceleration'] = np.where(inq_12 > 0, inq_6 / (inq_12 + 1e-8), 0)

        open_24 = df.get('acc_open_past_24mths', pd.Series(0, index=df.index)).fillna(0)
        open_6 = df.get('open_acc_6m', pd.Series(0, index=df.index)).fillna(0)
        df['open_acceleration'] = np.where(open_24 > 0, open_6 / (open_24 + 1e-8), 0)

        for c in ['issue_d', 'earliest_cr_line']:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

    if dataset_name == 'internet':
        drop_date_cols = ['earlies_credit_mon', 'issue_month', 'year_of_loan', 'issue_date',
                          'loan_id', 'user_id', 'title', 'policy_code',
                          'early_return', 'early_return_amount', 'early_return_amount_3mon']
        drop_date_exist = [c for c in drop_date_cols if c in df.columns]
        df = df.drop(columns=drop_date_exist)

    cat_features = df.select_dtypes(include=["object", "bool", "category"]).columns
    for feat in cat_features:
        if feat != target_col:
            df[feat] = df[feat].astype('category').cat.codes

    target = df[target_col].values.astype(np.int64)
    features = df.drop(columns=[target_col]).copy()

    for c in features.columns:
        features[c] = pd.to_numeric(features[c], errors='coerce')
    features = features.fillna(0)

    scaler = StandardScaler()
    features[features.columns] = scaler.fit_transform(features[features.columns])
    print(f'Standardized features: {features.shape[1]}')

    return features.values.astype(np.float32), target


# ============================================================
# 2. RCSMOTE
# ============================================================
def rcsmote(X, y, k=5, random_state=42):
    rng = np.random.RandomState(random_state)
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_diff = len(X_maj) - len(X_min)
    if n_diff <= 0 or len(X_min) < 2:
        return X, y

    lo = X_min.min(axis=0)
    hi = X_min.max(axis=0)

    k_use = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)

    synth = []
    for _ in range(n_diff):
        i = rng.randint(0, len(X_min))
        j = idx[i, rng.randint(1, k_use + 1)]
        lam = rng.rand()
        s = X_min[i] + lam * (X_min[j] - X_min[i])
        s = np.clip(s, lo, hi)
        synth.append(s)
    synth = np.array(synth, dtype=np.float32)
    X_new = np.vstack([X, synth])
    y_new = np.concatenate([y, np.ones(len(synth), dtype=np.int64)])
    perm = rng.permutation(len(X_new))
    return X_new[perm], y_new[perm]


# ============================================================
# 3. Lightweight Bi-LSTM with neuron pruning (Eq.6, Eq.7)
# ============================================================
class LightBiLSTM(nn.Module):
    """
    Lightweight Bi-LSTM with weight-norm based neuron pruning.
    - L1/L2 regularization for weight sparsity (Eq.7)
    - Neuron pruning by sorting hidden weight norms (Eq.6)
    """
    def __init__(self, input_dim, hidden=16, prune_ratio=0.3, l1_lambda=1e-5):
        super().__init__()
        self.hidden = hidden
        self.prune_ratio = prune_ratio
        self.l1_lambda = l1_lambda
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

        # Pruning mask: keep top-k neurons based on weight norm (Eq.6)
        self.register_buffer('neuron_mask', torch.ones(hidden * 2))

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, F) -> (B, 1, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = out * self.neuron_mask.to(out.device)
        return torch.sigmoid(self.fc(out)).squeeze(-1)

    def compute_reg_loss(self):
        """
        L1 regularization term (Eq.7) promoting weight sparsity.
        """
        reg = 0.0
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                reg = reg + p.abs().sum()
        return self.l1_lambda * reg

    @torch.no_grad()
    def prune_neurons(self):
        """
        Neuron pruning (Eq.6):
        1. Compute L2 norm of output weights for each hidden neuron.
        2. Sort and mask the lowest prune_ratio fraction.
        """
        importance = self.fc.weight.data.abs().squeeze(0)  # (hidden*2,)
        k = int(len(importance) * (1 - self.prune_ratio))
        if k < 1:
            k = 1
        threshold = torch.topk(importance, k, largest=True).values.min()
        new_mask = (importance >= threshold).float()
        self.neuron_mask.copy_(new_mask)


def train_bilstm(X_tr, y_tr, X_va, y_va, epochs=3, hidden=32, lr=3e-1, bs=1024,
                 use_pruning=True):
    model = LightBiLSTM(X_tr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            p = model(Xt[idx])
            loss = loss_fn(p, yt[idx]) + model.compute_reg_loss()  # L1 regularization
            loss.backward()
            opt.step()
        if use_pruning and ep == epochs - 1:
            model.prune_neurons()

    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy()
    try:
        auc = roc_auc_score(y_va, pv)
    except Exception:
        auc = 0.5
    return model, auc


# ============================================================
# 4. Parrot Optimization Algorithm (POA) - Eq.12 - Eq.15
# ============================================================
def levy_flight(dim, beta=1.5):
    """
    Mantegna's algorithm for Levy distribution step length.
    """
    sigma_u = (gamma_func(1 + beta) * math.sin(math.pi * beta / 2) /
               (gamma_func((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta) + 1e-10)
    return step


def parrot_optimization_fs(X_tr, y_tr, X_va, y_va, n_pop=6, n_iter=5):
    """
    Parrot Optimization Algorithm (POA) for feature selection.
    Implements four behaviors: Foraging (Eq.12), Staying (Eq.13),
    Communicating (Eq.14), and Fear of Strangers (Eq.15).
    """
    n_feat = X_tr.shape[1]
    # Continuous position vectors Y ∈ [0,1]^n_feat, thresholded to binary mask
    Y = np.random.rand(n_pop, n_feat)

    def to_mask(y_vec):
        mask = (y_vec > 0.5).astype(int)
        if mask.sum() == 0:
            mask[np.random.randint(n_feat)] = 1
        return mask

    def evaluate(y_vec):
        mask = to_mask(y_vec)
        cols = np.where(mask == 1)[0]
        _, auc = train_bilstm(X_tr[:, cols], y_tr, X_va[:, cols], y_va,
                              epochs=2, use_pruning=False)
        return 0.9 * auc + 0.1 * (1 - len(cols) / n_feat)

    fitness = np.array([evaluate(Y[i]) for i in range(n_pop)])
    best_idx = fitness.argmax()
    Y_best = Y[best_idx].copy()
    best_fit = fitness[best_idx]

    Max_iter = n_iter
    for t in range(n_iter):
        Y_mean = Y.mean(axis=0)

        for i in range(n_pop):
            St = np.random.randint(1, 5)  # randomly choose behavior 1-4
            Y_i = Y[i].copy()

            if St == 1:
                # Eq.12 Foraging
                L = levy_flight(n_feat)
                factor = (1 - t / Max_iter) ** (2 * t / Max_iter)
                Y_new = (Y_best - Y_i) * L + np.random.rand() * factor * Y_i

            elif St == 2:
                # Eq.13 Staying
                L = levy_flight(n_feat)
                Y_new = Y_i + Y_best * L

            elif St == 3:
                # Eq.14 Communicating
                Q = np.random.rand()
                if Q <= 0.5:
                    Y_new = 0.2 * (1 - t / Max_iter) * (Y_i - Y_mean)
                else:
                    Y_new = 0.2 * (1 - t / Max_iter) * np.exp(-t / (np.random.rand() * Max_iter + 1e-8))
                    Y_new = Y_new * np.ones(n_feat) + Y_i

            else:
                # Eq.15 Fear of Strangers
                term1 = np.random.rand() * math.cos(0.5 * math.pi * t / Max_iter) * (Y_best - Y_i)
                term2 = math.cos(np.random.rand() * math.pi) * (t / Max_iter) ** 2 * \
                        (2 * (Y_i - Y_best) / Max_iter)
                Y_new = Y_i + term1 - term2

            Y_new = np.clip(Y_new, 0, 1)
            fit_new = evaluate(Y_new)
            if fit_new > fitness[i]:
                Y[i] = Y_new
                fitness[i] = fit_new
                if fit_new > best_fit:
                    Y_best = Y_new.copy()
                    best_fit = fit_new
        print(f"  [POA] iter {t+1}/{n_iter}  best_fit={best_fit:.4f}  "
              f"n_feat={to_mask(Y_best).sum()}")

    return np.where(to_mask(Y_best) == 1)[0]


# ============================================================
# 5. Deep-Spiking Kepler NN (DSKNN) - Section 3.4.1
# ============================================================
class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for LIF spiking neuron.
    Forward: Heaviside step (spike firing)
    Backward: arctan surrogate gradient
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alpha = 2.0
        grad = grad_out * (1.0 / (1.0 + (math.pi * x * alpha) ** 2))
        return grad


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.
    Membrane potential: V_{t+1} = tau * V_t * (1-S_t) + I_t
    Spike: S_t = H(V_t - V_th)
    """
    def __init__(self, tau=0.5, v_th=1.0, T=4):
        super().__init__()
        self.tau = tau
        self.v_th = v_th
        self.T = T

    def forward(self, x):
        v = torch.zeros_like(x)
        spike_sum = torch.zeros_like(x)
        for _ in range(self.T):
            v = self.tau * v + x
            spike = SurrogateSpike.apply(v - self.v_th)
            v = v * (1 - spike)  # hard reset
            spike_sum = spike_sum + spike
        return spike_sum / self.T   # rate coding


class DSKNN(nn.Module):
    """
    Deep-Spiking Kepler NN (Eq.16 & Eq.17):
    Conv1D -> BN -> AvgPool -> Dropout -> LIF -> FC -> LIF -> FC(out)
    """
    def __init__(self, input_dim, conv_ch=16, hidden=32, T=4,
                 tau=0.5, v_th=1.0, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(1, conv_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_ch)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)

        pooled_dim = input_dim // 2

        self.fc1 = nn.Linear(conv_ch * pooled_dim, hidden)
        self.lif1 = LIFNeuron(tau=tau, v_th=v_th, T=T)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.lif2 = LIFNeuron(tau=tau, v_th=v_th, T=T)
        self.fc_out = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, F) -> (B, 1, F)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool(h)
        h = self.dropout1(h)
        h = h.flatten(1)
        h = self.fc1(h)
        h = self.lif1(h)
        h = self.dropout2(h)
        h = self.fc2(h)
        h = self.lif2(h)
        out = self.fc_out(h)
        return torch.sigmoid(out).squeeze(-1)


# ============================================================
# 6. Kepler Optimization Algorithm (KOA) - Section 3.4.2
# ============================================================
def kepler_optimization_algorithm(fitness_fn, dim, bounds, n_pop=5, n_iter=5):
    """
    Kepler Optimization Algorithm.
    Implements Eq.19 - Eq.25.

    Args:
        fitness_fn: fitness function f(x) -> scalar (maximization)
        dim: dimension of decision variables
        bounds: [(lo, hi), ...] for each dimension
        n_pop, n_iter: population size and iterations
    Returns:
        best_x, best_fit
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Eq.19: Initialize planet positions
    X = lo + np.random.rand(n_pop, dim) * (hi - lo)
    # Physical properties: mass, eccentricity, semi-major axis
    m = np.random.rand(n_pop)
    e = np.random.rand(n_pop)
    a = np.random.rand(n_pop) * 2 + 1

    fitness = np.array([fitness_fn(X[i]) for i in range(n_pop)])
    sun_idx = fitness.argmax()
    X_sun = X[sun_idx].copy()
    best_fit = fitness[sun_idx]

    G = 6.6743e-11 * 1e10  # scaled gravitational constant

    for t in range(n_iter):
        R = np.linalg.norm(X - X_sun, axis=1) + 1e-10
        R_norm = R / (R.max() + 1e-10)

        for i in range(n_pop):
            if i == sun_idx:
                continue

            # Eq.21: Gravitational force
            M_sun = 1.0
            F_grav = G * M_sun * m[i] / (R[i] ** 2 + 1e-10)

            # Eq.22: Velocity based on S_j norm branch
            S_j = X[i] - X_sun
            S_norm = np.linalg.norm(S_j) / (np.linalg.norm(hi - lo) + 1e-10)

            r1 = np.random.rand()
            if S_norm <= 0.5:
                # Perihelion: high-speed orbital motion
                V_i = r1 * np.sqrt(F_grav * (1 + e[i]) / (a[i] * (1 - e[i]) + 1e-10)) * \
                      (X_sun - X[i]) / (np.linalg.norm(X_sun - X[i]) + 1e-10)
            else:
                # Aphelion: low-speed drift
                V_i = r1 * np.sqrt(F_grav * (1 - e[i]) / (a[i] * (1 + e[i]) + 1e-10)) * \
                      (np.random.randn(dim))

            # Eq.23: Update position
            r2 = np.random.rand()
            X_new = X[i] + V_i + r2 * (X_sun - X[i])

            # Eq.24: Adaptive distance adjustment
            h_factor = (1 - t / n_iter) * np.random.randn(dim)
            X_new = X_new + h_factor * (X_sun - X[i]) * 0.05

            X_new = np.clip(X_new, lo, hi)
            fit_new = fitness_fn(X_new)

            # Eq.25: Elitism
            if fit_new > fitness[i]:
                X[i] = X_new
                fitness[i] = fit_new
                if fit_new > best_fit:
                    best_fit = fit_new
                    X_sun = X_new.copy()
                    sun_idx = i

        print(f"  [KOA] iter {t+1}/{n_iter}  best_fit={best_fit:.4f}  "
              f"sun_pos={np.round(X_sun, 3)}")

    return X_sun, best_fit


# ============================================================
# 7. DSKNN training function (with epoch timing)
# ============================================================
def train_dsknn(X_tr, y_tr, X_va, y_va, hyperparams, epochs=15, bs=512,
                verbose=True, return_epoch_time=False):
    """
    hyperparams: dict {hidden, tau, v_th, lr, dropout}
    """
    hidden  = int(hyperparams['hidden'])
    tau     = float(hyperparams['tau'])
    v_th    = float(hyperparams['v_th'])
    lr      = float(hyperparams['lr'])
    dropout = float(hyperparams['dropout'])

    model = DSKNN(X_tr.shape[1], conv_ch=16, hidden=hidden, T=4,
                  tau=tau, v_th=v_th, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)

    epoch_times = []
    model.train()
    for ep in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()

        perm = torch.randperm(len(Xt))
        total_loss = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            p = model(Xt[idx])
            loss = loss_fn(p, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()
        epoch_times.append(t_end - t_start)

        if verbose:
            print(f"  [DSKNN] Epoch {ep+1}/{epochs}  "
                  f"loss={total_loss:.4f}  time={epoch_times[-1]:.3f}s")

    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy()
    try:
        auc = roc_auc_score(y_va, pv)
    except Exception:
        auc = 0.5

    avg_epoch_time = np.mean(epoch_times)
    if return_epoch_time:
        return model, auc, avg_epoch_time
    return model, auc


# ============================================================
# 8. Evaluation metrics
# ============================================================
def compute_ks(y_true, y_score):
    return ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


# ============================================================
# 9. Main pipeline
# ============================================================
def main():
    # ---- Data loading ----
    X, y = load_and_clean(DATASET)
    print(f"Data shape: {X.shape}, positive rate: {y.mean():.4f}")

    # ---- Split ----
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.2, random_state=42, stratify=y_trv)

    # ---- RCSMOTE ----
    print(">> RCSMOTE oversampling ...")
    X_tr_bal, y_tr_bal = rcsmote(X_tr, y_tr, k=5)
    print(f"Oversampled training set: {X_tr_bal.shape}, positive rate: {y_tr_bal.mean():.4f}")

    # ---- POA + Lightweight Bi-LSTM feature selection ----
    with timer("POA-BiLSTM Feature Selection"):
        print(">> POA-BiLSTM feature selection ...")
        sel_idx = parrot_optimization_fs(X_tr_bal, y_tr_bal, X_va, y_va,
                                         n_pop=6, n_iter=5)
    print(f"Selected features: {len(sel_idx)} / {X_tr.shape[1]}")

    X_tr_sel = X_tr_bal[:, sel_idx]
    X_va_sel = X_va[:, sel_idx]
    X_te_sel = X_te[:, sel_idx]

    # ---- KOA hyperparameter optimization for DSKNN ----
    with timer("KOA Hyperparameter Optimization for DSKNN"):
        print(">> KOA optimizing DSKNN hyperparameters ...")

        # Search space: [hidden, tau, v_th, lr, dropout]
        bounds = [
            (16, 64),        # hidden
            (0.3, 0.9),      # tau
            (0.5, 1.5),      # v_th
            (1e-4, 5e-3),    # lr
            (0.1, 0.5),      # dropout
        ]

        def koa_fitness(x):
            hp = {'hidden': int(x[0]), 'tau': x[1], 'v_th': x[2],
                  'lr': x[3], 'dropout': x[4]}
            _, auc = train_dsknn(X_tr_sel, y_tr_bal, X_va_sel, y_va,
                                 hyperparams=hp, epochs=3, verbose=False)
            return auc

        best_x, best_koa_fit = kepler_optimization_algorithm(
            koa_fitness, dim=5, bounds=bounds, n_pop=5, n_iter=4)
        best_hp = {'hidden': int(best_x[0]), 'tau': best_x[1],
                   'v_th': best_x[2], 'lr': best_x[3], 'dropout': best_x[4]}
        print(f"KOA best hyperparams: {best_hp}, val_auc={best_koa_fit:.4f}")

    # ---- Final DSKNN training ----
    with timer("Final DSKNN Training"):
        print(">> Final training of DSKNN ...")
        X_full = np.vstack([X_tr_sel, X_va_sel])
        y_full = np.concatenate([y_tr_bal, y_va])
        model, _, avg_epoch_time = train_dsknn(
            X_full, y_full, X_te_sel, y_te,
            hyperparams=best_hp, epochs=15, bs=512, verbose=True,
            return_epoch_time=True)

    # ---- Test set prediction ----
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_te_sel, dtype=torch.float32).to(device)
        y_prob = model(Xte_t).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    # ---- Metrics ----
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob)
    gini = 2 * auc - 1
    ks   = compute_ks(y_te, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\n" + "=" * 55)
    print(f"    Test Set Evaluation (LPBiLSTM + DSKNN + KOA) - Dataset: {DATASET}")
    print("=" * 55)
    print(f"{'AUC':<15} {auc:>10.4f}")
    print(f"{'Accuracy':<15} {acc:>10.4f}")
    print(f"{'Precision':<15} {prec:>10.4f}")
    print(f"{'Recall':<15} {rec:>10.4f}")
    print(f"{'F1':<15} {f1:>10.4f}")
    print(f"{'Gini':<15} {gini:>10.4f}")
    print(f"{'KS':<15} {ks:>10.4f}")
    print(f"{'Type II':<15} {type2:>10.4f}")
    print(f"Average Training Time per Epoch: {avg_epoch_time:.4f} seconds")
    print("=" * 55)


if __name__ == '__main__':
    main()