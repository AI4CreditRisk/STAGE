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
DATASET = 'lendingclub'   # Options: 'lendingclub' | 'gmsc' | 'internet' | 'creditcard'

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
        df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in default_set else 0)

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

    if 'int_rate' in df.columns:
        df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'revol_util' in df.columns:
        df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    if 'emp_length' in df.columns:
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
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

else:
    raise ValueError("Unknown DATASET")

# ==================================================================
# Unified preprocessing
# ==================================================================
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
for c in cat_cols:
    if c != target_col:
        df[c] = df[c].astype('category').cat.codes

target = df[target_col].values.astype(np.int64)
features = df.drop(columns=[target_col]).copy()

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

# ==================================================================
# Split dataset
# ==================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    features.values,
    target,
    test_size=0.2,
    random_state=SEED,
    stratify=target
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp
)

# Convert to tensors, add sequence dimension for LSTM
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ==================================================================
# 2. Model: LSTM + Dropout
# ==================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)

        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# ==================================================================
# 3. Train
# ==================================================================
def train_lstm_classifier(model, dataloader, val_dataloader, optimizer, class_weights, epochs=20):
    train_losses = []
    val_losses = []
    epoch_times = []

    criterion = nn.BCELoss(reduction='none')

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        for data, target in dataloader:
            optimizer.zero_grad()

            pred_y = model(data)

            weights = class_weights[target.long().squeeze()]
            weights = weights.view(-1, 1)

            loss = criterion(pred_y, target)
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        avg_train_loss = train_loss / len(dataloader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_dataloader:
                pred_y = model(data)

                weights = class_weights[target.long().squeeze()]
                weights = weights.view(-1, 1)

                loss = criterion(pred_y, target)
                loss = (loss * weights).mean()

                val_loss += loss.item() * data.size(0)

        avg_val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss:   {avg_val_loss:.6f}')
        print(f'Epoch Time: {epoch_time:.2f}s')
        print('-' * 40)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f'Average Training Time per Epoch: {avg_epoch_time:.2f}s')

    return train_losses, val_losses

# ==================================================================
# 4. Hyperparameters
# ==================================================================
input_dim = X_train.shape[1]

hidden_dim = 128
output_dim = 1
num_layers = 1
dropout_rate = 0.3

lstm_classifier = LSTMClassifier(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    dropout_rate=dropout_rate
)

optimizer = optim.Adam(
    lstm_classifier.parameters(),
    lr=0.001,
    weight_decay=1e-2
)

# ==================================================================
# 5. Train model
# ==================================================================
train_losses, val_losses = train_lstm_classifier(
    model=lstm_classifier,
    dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    class_weights=class_weights,
    epochs=20
)

# ==================================================================
# 6. Evaluate
# ==================================================================
lstm_classifier.eval()
with torch.no_grad():
    pred_y_test = lstm_classifier(X_test_tensor).numpy().squeeze()

y_pred = (pred_y_test > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
gini = 2 * roc_auc - 1


def ks_statistic(y_true, y_prob):
    data = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })

    data = data.sort_values(by='y_prob', ascending=False)

    total_good = (data['y_true'] == 0).sum()
    total_bad = (data['y_true'] == 1).sum()

    data['cumulative_good'] = (data['y_true'] == 0).cumsum() / total_good
    data['cumulative_bad'] = (data['y_true'] == 1).cumsum() / total_bad

    data['ks'] = np.abs(data['cumulative_bad'] - data['cumulative_good'])

    return data['ks'].max()


ks_stat = ks_statistic(y_test, pred_y_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
type_II_error = fn / (fn + tp) if (fn + tp) > 0 else 0

print('LSTM: ##############')
print(f'ROC AUC: {roc_auc}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Gini Coefficient: {gini}')
print(f'KS Statistic: {ks_stat}')
print(f'Type II Error: {type_II_error}')
print(roc_auc, accuracy, precision, recall, f1, gini, ks_stat)