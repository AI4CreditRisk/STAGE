import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
DATASET = 'lendingclub'    # Options: 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

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

X_train, X_temp, y_train, y_temp = train_test_split(features.values, target, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# Convert to PyTorch tensors and DataLoaders
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ==================================================================
# 2. Model
# ==================================================================
class DGPGVAEClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(DGPGVAEClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean and log-variance
        )
        self.gated_prior_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # gated prior mean and log-variance
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
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def gated_prior(self, x):
        gp = self.gated_prior_network(x)
        gp_mu, gp_logvar = gp.chunk(2, dim=-1)
        return gp_mu, gp_logvar

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
        gp_mu, gp_logvar = self.gated_prior(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), self.classify(z), mu, logvar, gp_mu, gp_logvar

# ==================================================================
# 3. Train
# ==================================================================
def loss_function(recon_x, x, pred_y, y, mu, logvar, gp_mu, gp_logvar, class_weights, alpha):
    logvar    = torch.clamp(logvar,    min=-6, max=6)
    gp_logvar = torch.clamp(gp_logvar, min=-6, max=6)
    pred_y    = torch.clamp(pred_y, 1e-7, 1 - 1e-7)
    BCE = nn.functional.binary_cross_entropy(pred_y, y, weight=class_weights[y.long()].view(-1, 1), reduction='sum')
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - gp_mu.pow(2) - gp_logvar.exp() - (mu - gp_mu).pow(2) / gp_logvar.exp() - logvar.exp() / gp_logvar.exp())
    return MSE + KLD + alpha * BCE

def train_dgpgvae_classifier(model, dataloader, val_dataloader, optimizer, class_weights, alpha, epochs=20):
    model.train()
    train_losses = []
    val_losses = []
    epoch_times = []  # Record time for each epoch
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            recon_batch, pred_y, mu, logvar, gp_mu, gp_logvar = model(data)
            loss = loss_function(recon_batch, data, pred_y, target, mu, logvar, gp_mu, gp_logvar, class_weights, alpha)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_losses.append(train_loss / len(dataloader.dataset))
        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_dataloader:
                recon_batch, pred_y, mu, logvar, gp_mu, gp_logvar = model(data)
                loss = loss_function(recon_batch, data, pred_y, target, mu, logvar, gp_mu, gp_logvar, class_weights, alpha)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataloader.dataset))
        print(f'Epoch {epoch + 1}, Val Loss: {val_losses[-1]}')

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f'Average Training Time per Epoch: {avg_epoch_time:.2f}s')
    return train_losses, val_losses

# Hyperparameters
input_dim = X_train.shape[1]
latent_dim = 8
output_dim = 1
alpha = 1.0

# Initialize and train the DG-PG-VAE Classifier
dgpgvae_classifier = DGPGVAEClassifier(input_dim, latent_dim, output_dim)
optimizer = optim.Adam(dgpgvae_classifier.parameters(), lr=0.001)

train_losses, val_losses = train_dgpgvae_classifier(
    dgpgvae_classifier, train_dataloader, val_dataloader, optimizer, class_weights, alpha
)

# ==================================================================
# 4. Evaluate
# ==================================================================
with torch.no_grad():
    mu_test, logvar_test = dgpgvae_classifier.encode(X_test_tensor)
    latent_test = dgpgvae_classifier.reparameterize(mu_test, logvar_test)
    pred_y_test = dgpgvae_classifier.classify(latent_test).numpy().squeeze()

y_pred = (pred_y_test > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
gini = 2 * roc_auc - 1

def ks_statistic(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    data = data.sort_values(by='y_prob', ascending=False)
    data['cumulative_good'] = (data['y_true'] == 0).cumsum() / (data['y_true'] == 0).sum()
    data['cumulative_bad'] = (data['y_true'] == 1).cumsum() / (data['y_true'] == 1).sum()
    data['ks'] = np.abs(data['cumulative_bad'] - data['cumulative_good'])
    return data['ks'].max()

ks_stat = ks_statistic(y_test, pred_y_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
type_II_error = fn / (fn + tp)

print(f'ROC AUC: {roc_auc}')
print(f'Accucracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Gini Coefficient: {gini}')
print(f'KS Statistic: {ks_stat}')
print(f'Type II Error (False Negative Rate): {type_II_error}')
print(roc_auc, accuracy, precision, recall, f1, gini, ks_stat)