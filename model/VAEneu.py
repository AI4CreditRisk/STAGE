import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
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
DATASET = 'creditcard'   # Options: 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

# ----- Dataset 1: lendingclub -----
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

    target_col = 'loan_status'

# ----- Dataset 2: gmsc -----
elif DATASET == 'gmsc':
    input_csv = os.path.join(data_dir, "cs_training_fe_01.csv")
    df = pd.read_csv(input_csv)
    target_col = 'SeriousDlqin2yrs'

# ----- Dataset 3: internet -----
elif DATASET == 'internet':
    input_csv = os.path.join(data_dir, "train_internetdata3.csv")
    df = pd.read_csv(input_csv, low_memory=False)

    target_col = 'is_default'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

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
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
for c in cat_cols:
    if c != target_col:
        df[c] = df[c].astype('category').cat.codes

target = df[target_col].values.astype(np.int64)
features = df.drop(columns=[target_col]).copy()

# Clean inf/nan
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(0)
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')
features = features.fillna(0)

# Standardization
scaler = StandardScaler()
features[features.columns] = scaler.fit_transform(features[features.columns])

# Correlation filtering
correlation_matrix = features.corr()
correlation_threshold = 0.6
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            highly_correlated_features.add(correlation_matrix.columns[i])
selected_features = [f for f in features.columns if f not in highly_correlated_features]
features = features[selected_features]

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Class weights (handle class imbalance)
class_counts = np.bincount(y_train.astype(int))
total_samples = len(y_train)
weights = total_samples / (2 * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor.long())
val_dataset = TensorDataset(X_val_tensor, y_val_tensor.long())
test_dataset = TensorDataset(X_test_tensor, y_test_tensor.long())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==================================================================
# 2. Model
# ==================================================================
class VAEneu(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128, output_dim=1, n_steps_ahead=5):
        super(VAEneu, self).__init__()
        self.n_steps_ahead = n_steps_ahead
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # Deterministic path
        self.deterministic_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder / Forecaster
        self.forecaster = nn.GRU(hidden_dim + latent_dim, hidden_dim, batch_first=True)

        # Output heads (for each step ahead)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, deterministic_out):
        batch_size = z.size(0)
        z_repeated = z.unsqueeze(1).repeat(1, self.n_steps_ahead, 1)
        deterministic_out_expanded = deterministic_out.unsqueeze(1).repeat(1, self.n_steps_ahead, 1)
        combined_input = torch.cat([z_repeated, deterministic_out_expanded], dim=-1)
        gru_out, _ = self.forecaster(combined_input)
        mu = self.mu_head(gru_out)
        sigma = torch.nn.functional.softplus(self.sigma_head(gru_out))
        return mu, sigma

    def classify(self, z):
        return self.classifier(z).squeeze()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        deterministic_out = self.deterministic_path(x)
        mu_pred, sigma_pred = self.decode(z, deterministic_out)
        pred_y = self.classify(z)
        return mu_pred, sigma_pred, pred_y, mu, logvar

# ==================================================================
# 3. Train
# ==================================================================
def loss_function(mu_pred, sigma_pred, pred_y, y_true, mu, logvar, beta=1.0, class_weights=None):
    y_true_expanded = y_true.unsqueeze(1).unsqueeze(2).float().expand_as(mu_pred)
    dist = torch.distributions.Normal(mu_pred, sigma_pred)
    nll = -dist.log_prob(y_true_expanded).mean()
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    if class_weights is not None:
        bce = nn.functional.binary_cross_entropy(pred_y, y_true.float(), weight=class_weights[y_true.long()], reduction='mean')
    else:
        bce = nn.functional.binary_cross_entropy(pred_y, y_true.float(), reduction='mean')

    return nll + beta * KLD + bce

def train_vae_neu(model, train_loader, val_loader, optimizer, class_weights=None, beta=1.0, epochs=20):
    model.train()
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            mu_pred, sigma_pred, pred_y, mu, logvar = model(data)
            loss = loss_function(mu_pred, sigma_pred, pred_y, target, mu, logvar, beta, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader.dataset))
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                mu_pred, sigma_pred, pred_y, mu, logvar = model(data)
                loss = loss_function(mu_pred, sigma_pred, pred_y, target, mu, logvar, beta, class_weights)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader.dataset))
        print(f'Epoch {epoch+1}, Val Loss: {val_losses[-1]}')

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f'Average Training Time per Epoch: {avg_epoch_time:.2f}s')
    return train_losses, val_losses

# Hyperparameters
input_dim = X_train.shape[1]
latent_dim = 16
output_dim = 1
n_steps_ahead = 2
beta = 1.0

model = VAEneu(input_dim, latent_dim, output_dim=output_dim, n_steps_ahead=n_steps_ahead)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = train_vae_neu(model, train_loader, val_loader, optimizer,
                                         class_weights=class_weights, beta=beta, epochs=20)

# ==================================================================
# 4. Evaluate
# ==================================================================
with torch.no_grad():
    mu_pred_test, sigma_pred_test, pred_y_test, _, _ = model(X_test_tensor)

pred_scores = pred_y_test.numpy()
pred_labels = (pred_scores > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, pred_scores)
accuracy = accuracy_score(y_test, pred_labels)
precision = precision_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels)
f1 = f1_score(y_test, pred_labels)
gini = 2 * roc_auc - 1

def ks_statistic(y_true, y_scores):
    data = pd.DataFrame({'y': y_true, 'score': y_scores})
    data = data.sort_values(by='score', ascending=False).reset_index(drop=True)
    data['cum_good'] = (data['y'] == 0).cumsum() / ((data['y'] == 0).sum())
    data['cum_bad'] = (data['y'] == 1).cumsum() / ((data['y'] == 1).sum())
    data['ks'] = np.abs(data['cum_bad'] - data['cum_good'])
    return data['ks'].max()

ks = ks_statistic(y_test, pred_scores)

tn, fp, fn, tp = confusion_matrix(y_test, pred_labels).ravel()
type_ii_error = fn / (fn + tp)

print(f'ROC AUC: {roc_auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Gini Coefficient: {gini:.4f}')
print(f'KS Statistic: {ks:.4f}')
print(f'Type II Error (False Negative Rate): {type_ii_error:.4f}')
print(roc_auc, accuracy, precision, recall, f1, gini, ks)