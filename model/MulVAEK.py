import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
# ==============================
# Fix random seeds for reproducibility
# ==============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ==================================================================
# 1. Load data
# ==================================================================
DATASET = 'creditcard'    # Options: 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

# ----- Dataset 1: lendingclub -----
if DATASET == 'lendingclub':
    input_csv = os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    target_col = 'loan_status'
    default_set = {'Charged Off', 'Default', 'Late (31-120 days)',
                   'In Grace Period', 'Late (16-30 days)'}
    good_set = {'Current', 'Fully Paid'}

    if pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(int)
    else:
        df[target_col] = df[target_col].astype(str).str.strip()
        valid_status = default_set.union(good_set)
        df = df[df[target_col].isin(valid_status)].copy()
        df[target_col] = df[target_col].apply(lambda x: 1 if x in default_set else 0)

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

    # LendingClub specific transformations
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

# ----- Dataset 2: gmsc -----
elif DATASET == 'gmsc':
    input_csv = os.path.join(data_dir, "cs_training_fe_01.csv")
    df = pd.read_csv(input_csv)
    target_col = 'SeriousDlqin2yrs'
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

# ----- Dataset 3: internet -----
elif DATASET == 'internet':
    input_csv = os.path.join(data_dir, "train_internetdata3.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    target_col = 'is_default'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    leakage_cols = [
        'loan_id', 'user_id', 'title', 'policy_code',
        'early_return', 'early_return_amount', 'early_return_amount_3mon',
    ]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

# ----- Dataset 4: creditcard -----
elif DATASET == 'creditcard':
    input_csv = os.path.join(data_dir, "default of credit card clientsUCI.csv")
    try:
        df = pd.read_excel(input_csv, skiprows=1)
    except Exception:
        df = pd.read_csv(input_csv, skiprows=1)
    target_col = 'default payment next month'
    if target_col not in df.columns:
        cand = [c for c in df.columns if 'default' in c.lower()]
        target_col = cand[0]
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

# ==================================================================
# Unified preprocessing: encoding, cleaning, scaling, correlation filtering
# ==================================================================
df = df.dropna(subset=[target_col]).copy()
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(int)
df = df[df[target_col].isin([0, 1])].copy()

cat_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
for c in cat_cols:
    if c != target_col:
        df[c] = df[c].astype('category').cat.codes

target = df[target_col].values.astype(np.int64)
features = df.drop(columns=[target_col]).copy()

# Clean inf/nan and ensure numeric
features = features.replace([np.inf, -np.inf], np.nan)
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')
features = features.fillna(0)

# Drop constant columns
nunique = features.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    features = features.drop(columns=constant_cols)

# Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

# Correlation filtering
correlation_threshold = 0.6
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
features = features.drop(columns=high_corr_features)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    features, target, test_size=0.2, random_state=SEED, stratify=target
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(dtype=np.float32))
X_val_tensor = torch.tensor(X_val.to_numpy(dtype=np.float32))
X_test_tensor = torch.tensor(X_test.to_numpy(dtype=np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.float32)).view(-1, 1)
y_val_tensor = torch.tensor(y_val.astype(np.float32)).view(-1, 1)
y_test_tensor = torch.tensor(y_test.astype(np.float32)).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

BATCH_SIZE = 128
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weights
class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

# ==================================================================
# 2. Model
# ==================================================================
class MulVAEKClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_clusters=10):
        super(MulVAEKClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mu + logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        # Cluster centers
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        pred_y = self.classify(z)
        return recon_x, pred_y, mu, logvar, z

# ==================================================================
# 3. Train
# ==================================================================
def mulvaek_loss(recon_x, x, pred_y, y, mu, logvar, z, model, class_weights, alpha, beta, gamma):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    y = y.float().view_as(pred_y)
    y_long = y.long().view(-1)
    sample_weights = class_weights[y_long].view_as(pred_y)

    bce = nn.functional.binary_cross_entropy(pred_y, y, weight=sample_weights, reduction='sum')

    # K-means clustering loss
    dist = torch.cdist(z, model.mu_c)
    cluster_loss = torch.min(dist, dim=1).values.sum()

    total_loss = recon_loss + kld * beta + bce * alpha + cluster_loss * gamma
    return total_loss, recon_loss, kld, bce, cluster_loss

def train_mulvaek(model, train_loader, val_loader, optimizer, class_weights,
                  epochs=20, alpha=1.0, beta=1.0, gamma=0.1):
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for data, target_batch in train_loader:
            optimizer.zero_grad()
            recon_batch, pred_y, mu, logvar, z = model(data)
            loss, _, _, _, _ = mulvaek_loss(
                recon_batch, data, pred_y, target_batch, mu, logvar, z,
                model, class_weights, alpha, beta, gamma
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target_batch in val_loader:
                recon_batch, pred_y, mu, logvar, z = model(data)
                loss, _, _, _, _ = mulvaek_loss(
                    recon_batch, data, pred_y, target_batch, mu, logvar, z,
                    model, class_weights, alpha, beta, gamma
                )
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader.dataset))

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f'Average Training Time per Epoch: {avg_epoch_time:.2f}s')
    return train_losses, val_losses

# Hyperparameters
input_dim = X_train.shape[1]
latent_dim = 4
output_dim = 1
n_clusters = 10
learning_rate = 0.001
alpha = 1.0
beta = 0.90   # tuned for creditcard dataset
gamma = 0.1

model = MulVAEKClassifier(input_dim, latent_dim, output_dim, n_clusters=n_clusters)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, val_losses = train_mulvaek(
    model, train_dataloader, val_dataloader, optimizer, class_weights,
    epochs=20, alpha=alpha, beta=beta, gamma=gamma
)

# ==================================================================
# 4. Evaluate
# ==================================================================
def ks_statistic(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    data = data.sort_values(by='y_prob', ascending=False)
    good_total = (data['y_true'] == 0).sum()
    bad_total = (data['y_true'] == 1).sum()
    data['cumulative_good'] = (data['y_true'] == 0).cumsum() / (good_total + 1e-8)
    data['cumulative_bad'] = (data['y_true'] == 1).cumsum() / (bad_total + 1e-8)
    data['ks'] = np.abs(data['cumulative_bad'] - data['cumulative_good'])
    return data['ks'].max()

model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for data, target_batch in test_dataloader:
        _, pred_y, _, _, _ = model(data)
        all_preds.extend(pred_y.cpu().numpy().flatten())
        all_true.extend(target_batch.cpu().numpy().flatten())

all_preds = np.array(all_preds)
all_true = np.array(all_true).astype(int)
y_pred = (all_preds > 0.5).astype(int)

roc_auc = roc_auc_score(all_true, all_preds)
accuracy = accuracy_score(all_true, y_pred)
precision = precision_score(all_true, y_pred, zero_division=0)
recall = recall_score(all_true, y_pred, zero_division=0)
f1 = f1_score(all_true, y_pred, zero_division=0)
gini = 2 * roc_auc - 1
ks = ks_statistic(all_true, all_preds)

cm = confusion_matrix(all_true, y_pred)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn, fp, fn, tp = 0, 0, 0, 0
type_ii_error = fn / (fn + tp + 1e-8)

print(f'GIB: ##############')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Gini: {gini:.4f}')
print(f'KS Stat: {ks:.4f}')
print(f'Type II Error: {type_ii_error:.4f}')
print(f'Confusion Matrix:')
print(cm)
print(roc_auc, accuracy, precision, recall, f1, gini, ks)