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

# 0. Random seed and device
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load data
DATASET = 'lendingclub'  # Options: 'lendingclub', 'gmsc', 'internet', 'creditcard'

if DATASET == 'lendingclub':
    input_csv = os.path.join(data_dir, "LoanStatus_2017Q1_ml_01.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    default_set = {'Charged Off', 'Default', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)'}
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
    leakage_cols = ['id','member_id','url','desc','emp_title','title','funded_amnt','funded_amnt_inv',
        'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
        'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
        'next_pymnt_d','last_credit_pull_d','pymnt_plan','initial_list_status','policy_code',
        'hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term',
        'hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date',
        'hardship_length','hardship_dpd','hardship_loan_status','orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount','hardship_last_payment_amount','debt_settlement_flag',
        'debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount',
        'settlement_percentage','settlement_term','disbursement_method','revol_bal_joint',
        'sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc',
        'sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts',
        'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med',
        'sec_app_mths_since_last_major_derog']
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    if 'int_rate' in df.columns:
        df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'revol_util' in df.columns:
        df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    if 'emp_length' in df.columns:
        emp_map = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,
                   '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}
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
    for col in ['issue_d', 'earliest_cr_line']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    target_col = 'loan_status'

elif DATASET == 'gmsc':
    input_csv = os.path.join(data_dir, "cs_training_fe_01.csv")
    df = pd.read_csv(input_csv)
    target_col = 'SeriousDlqin2yrs'

elif DATASET == 'internet':
    input_csv = os.path.join(data_dir, "train_internetdata3.csv")
    df = pd.read_csv(input_csv, low_memory=False)
    target_col = 'is_default'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    leakage_cols = ['loan_id','user_id','title','policy_code','early_return','early_return_amount','early_return_amount_3mon']
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

elif DATASET == 'creditcard':
    input_csv = os.path.join(data_dir, "default of credit card clientsUCI.csv")
    try:
        df = pd.read_excel(input_csv, skiprows=1)
    except Exception:
        df = pd.read_csv(input_csv, skiprows=1)
    target_col = 'default payment next month'
    if target_col not in df.columns:
        candidate_cols = [c for c in df.columns if 'default' in str(c).lower()]
        if len(candidate_cols) == 0:
            raise ValueError("Target column was not found.")
        target_col = candidate_cols[0]
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
else:
    raise ValueError("Unknown DATASET. Please select lendingclub, gmsc, internet or creditcard.")

# 2. Unified preprocessing
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' does not exist.")
df = df.dropna(subset=[target_col]).copy()
target = pd.to_numeric(df[target_col], errors='coerce')
valid_rows = target.notna()
df = df.loc[valid_rows].copy()
target = target.loc[valid_rows].astype(np.int64).values
cat_cols = df.select_dtypes(include=['object','bool','category']).columns.tolist()
for col in cat_cols:
    if col != target_col:
        df[col] = df[col].astype('category').cat.codes
features = df.drop(columns=[target_col]).copy()
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(0)
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')
features = features.fillna(0)
constant_cols = [col for col in features.columns if features[col].nunique() <= 1]
if len(constant_cols) > 0:
    features = features.drop(columns=constant_cols)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)
features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
correlation_matrix = features.corr()
correlation_threshold = 0.6
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > correlation_threshold:
            highly_correlated_features.add(correlation_matrix.columns[i])
selected_features = [col for col in features.columns if col not in highly_correlated_features]
features = features[selected_features]
print("Number of samples:", len(features))
print("Number of selected features:", features.shape[1])
print("Positive sample ratio:", target.mean())

# 3. Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(features.values, target, test_size=0.2,
                                                     random_state=SEED, stratify=target)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                random_state=SEED, stratify=y_temp)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# 4. Convert to PyTorch tensors (shape: [samples, 1, features])
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 5. DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 6. Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("Class weights:", class_weights.cpu().numpy())

# 7. Informer model
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.4):
        super(ProbSparseSelfAttention, self).__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        query_len = q.size(1)
        key_len = k.size(1)
        Q = self.q_linear(q).view(batch_size, query_len, self.nhead, self.head_dim)
        K = self.k_linear(k).view(batch_size, key_len, self.nhead, self.head_dim)
        V = self.v_linear(v).view(batch_size, key_len, self.nhead, self.head_dim)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / np.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, V)
        output = output.reshape(batch_size, query_len, self.d_model)
        output = self.out_linear(output)
        return output

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.4):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.4):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=1, num_layers=1, nhead=2, dropout=0.4):
        super(InformerClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = InformerEncoder(d_model=hidden_dim, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.encoder(x)
        x = x[:, -1, :]        # take last time step
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# 8. Training function
def train_informer_classifier(model, dataloader, val_dataloader, optimizer, class_weights, epochs=20):
    train_losses = []
    val_losses = []
    epoch_times = []
    criterion = nn.BCELoss(reduction='none')
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred_y = model(data)
            target_index = target.long().view(-1)
            weights = class_weights[target_index].view(-1, 1)
            loss = (criterion(pred_y, target) * weights).mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * data.size(0)
        avg_train_loss = train_loss_sum / len(dataloader.dataset)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                pred_y = model(data)
                target_index = target.long().view(-1)
                weights = class_weights[target_index].view(-1, 1)
                loss = (criterion(pred_y, target) * weights).mean()
                val_loss_sum += loss.item() * data.size(0)
        avg_val_loss = val_loss_sum / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss:   {avg_val_loss:.6f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")
    return train_losses, val_losses

# 9. Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 24
output_dim = 1
seq_len = 1
num_layers = 1
nhead = 2
dropout = 0.3
epochs = 20
learning_rate = 1e-3
weight_decay = 1e-3

# 10. Initialize model and optimizer
informer_classifier = InformerClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                         seq_len=seq_len, num_layers=num_layers, nhead=nhead, dropout=dropout).to(device)
optimizer = optim.Adam(informer_classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 11. Train model
train_losses, val_losses = train_informer_classifier(model=informer_classifier, dataloader=train_dataloader,
                                                     val_dataloader=val_dataloader, optimizer=optimizer,
                                                     class_weights=class_weights, epochs=epochs)

# 12. Evaluate model
informer_classifier.eval()
all_probs = []
with torch.no_grad():
    for data, target in test_dataloader:
        data = data.to(device)
        pred = informer_classifier(data)
        all_probs.append(pred.cpu().numpy())
pred_y_test = np.concatenate(all_probs, axis=0).squeeze()
pred_y_test = np.asarray(pred_y_test).reshape(-1)
y_pred = (pred_y_test > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
gini = 2 * roc_auc - 1

def ks_statistic(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    data = data.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
    total_good = (data['y_true'] == 0).sum()
    total_bad = (data['y_true'] == 1).sum()
    if total_good == 0 or total_bad == 0:
        return np.nan
    data['cumulative_good'] = (data['y_true'] == 0).cumsum() / total_good
    data['cumulative_bad'] = (data['y_true'] == 1).cumsum() / total_bad
    data['ks'] = np.abs(data['cumulative_bad'] - data['cumulative_good'])
    return data['ks'].max()

ks_stat = ks_statistic(y_test, pred_y_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
type_II_error = fn / (fn + tp) if (fn + tp) > 0 else 0

print("\n" + "=" * 60)
print("Informer with Mild Regularization")
print("=" * 60)
print(f"ROC AUC: {roc_auc:.6f}")
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1:.6f}")
print(f"Gini Coefficient: {gini:.6f}")
print(f"KS Statistic: {ks_stat:.6f}")
print(f"Type II Error: {type_II_error:.6f}")
print("\nConfusion Matrix:")
print(cm)
print("\nMetrics:")
print(roc_auc, accuracy, precision, recall, f1, gini, ks_stat)