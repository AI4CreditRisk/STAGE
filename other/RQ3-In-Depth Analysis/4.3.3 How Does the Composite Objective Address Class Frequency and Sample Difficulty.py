import os
import time
import copy
import math
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
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = 'gmsc'

BATCH_SIZE = 256
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-3

FOCAL_ALPHA_POS = None
FOCAL_ALPHA_NEG = None
FOCAL_GAMMA = 1.0
GATE_ENTROPY_LAMBDA = 0.001

PATIENCE = 10
MIN_DELTA = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(SCRIPT_DIR, 'data')):
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
elif os.path.exists(os.path.join(SCRIPT_DIR, '..', 'data')):
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
elif os.path.exists(os.path.join(SCRIPT_DIR, 'Dataset')):
    DATA_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
elif os.path.exists(os.path.join(SCRIPT_DIR, '..', 'Dataset')):
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Dataset'))
else:
    DATA_DIR = 'data'

print(f"Data directory resolved to: {DATA_DIR}")

FILE_MAP = {
    "lendingclub": os.path.join(DATA_DIR, "LoanStatus_2017Q1_ml_01.csv"),
    "gmsc": os.path.join(DATA_DIR, "cs_training_fe_01.csv"),
    "internet": os.path.join(DATA_DIR, "train_internetdata3.csv"),
    "creditcard": os.path.join(DATA_DIR, "default of credit card clientsUCI.csv"),
}

def load_dataset(dataset_name):
    if dataset_name not in FILE_MAP:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    input_file = FILE_MAP[dataset_name]

    if dataset_name == 'lendingclub':
        df = pd.read_csv(input_file, low_memory=False)
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
            'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
            'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
            'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
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

    elif dataset_name == 'gmsc':
        df = pd.read_csv(input_file, low_memory=False)
        for col in ['Unnamed: 0', 'ID', 'id']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        target_col = 'SeriousDlqin2yrs'
        if target_col not in df.columns:
            alt = [c for c in df.columns if c.lower() == 'seriousdlqin2yrs']
            if len(alt) > 0:
                df.rename(columns={alt[0]: target_col}, inplace=True)
            else:
                raise ValueError(f"未找到目标列 {target_col}")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    elif dataset_name == 'internet':
        df = pd.read_csv(input_file, low_memory=False)
        target_col = 'is_default'
        if target_col not in df.columns:
            raise ValueError(f"数据集中未找到目标列 '{target_col}'")
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(np.int64)
        leakage_cols = [
            'loan_id', 'user_id', 'title', 'policy_code',
            'early_return', 'early_return_amount', 'early_return_amount_3mon'
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    else:
        try:
            df = pd.read_excel(input_file, skiprows=1)
        except Exception:
            df = pd.read_csv(input_file, skiprows=1)
        target_col = 'default payment next month'
        if target_col not in df.columns:
            cand = [c for c in df.columns if 'default' in c.lower()]
            if len(cand) == 0:
                raise ValueError("未找到 default 标签列")
            target_col = cand[0]
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    if dataset_name == 'lendingclub':
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
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        for c in cat_cols:
            if c != target_col:
                df[c] = df[c].astype('category').cat.codes
        target = df[target_col].values.astype(np.int64)
        df = df.drop(columns=[target_col]).fillna(0)

    elif dataset_name == 'gmsc':
        for c in df.columns:
            if c != target_col:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.fillna(0)
        rename_map = {
            'RevolvingUtilizationOfUnsecuredLines': 'revolving_util',
            'age': 'age',
            'NumberOfTime30-59DaysPastDueNotWorse': 'n30_59',
            'DebtRatio': 'debt_ratio',
            'MonthlyIncome': 'monthly_income',
            'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
            'NumberOfTimes90DaysLate': 'n90_late',
            'NumberRealEstateLoansOrLines': 'real_estate_loans',
            'NumberOfTime60-89DaysPastDueNotWorse': 'n60_89',
            'NumberOfDependents': 'n_dependents'
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        target = df[target_col].values.astype(np.int64)
        df = df.drop(columns=[target_col]).fillna(0)
        if all(c in df.columns for c in ['monthly_income', 'debt_ratio']):
            df['debt_amount_est'] = df['monthly_income'] * df['debt_ratio']
        else:
            df['debt_amount_est'] = 0.0
        if all(c in df.columns for c in ['n30_59', 'n60_89', 'n90_late']):
            df['delinq_total'] = df['n30_59'] + df['n60_89'] + df['n90_late']
            df['delinq_severity'] = 1.0 * df['n30_59'] + 2.0 * df['n60_89'] + 3.0 * df['n90_late']
        else:
            df['delinq_total'] = 0.0
            df['delinq_severity'] = 0.0
        if all(c in df.columns for c in ['open_credit_lines', 'real_estate_loans']):
            df['credit_mix'] = df['open_credit_lines'] + df['real_estate_loans']
        else:
            df['credit_mix'] = 0.0
        if all(c in df.columns for c in ['monthly_income', 'n_dependents']):
            df['income_per_dep'] = df['monthly_income'] / (df['n_dependents'] + 1.0)
        else:
            df['income_per_dep'] = 0.0
        for col in ['n30_59', 'n60_89', 'n90_late']:
            if col in df.columns:
                df[f'{col}_flag'] = (df[col] > 0).astype(float)
            else:
                df[f'{col}_flag'] = 0.0

    elif dataset_name == 'internet':
        if 'interest' in df.columns:
            df['interest'] = pd.to_numeric(df['interest'].astype(str).str.replace('%', '', regex=False), errors='coerce')
        if 'recircle_util' in df.columns:
            df['recircle_util'] = pd.to_numeric(df['recircle_util'].astype(str).str.replace('%', '', regex=False), errors='coerce')
        if 'work_year' in df.columns:
            emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                       '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                       '8 years': 8, '9 years': 9, '10+ years': 10}
            df['work_year'] = df['work_year'].map(emp_map)
        if 'earlies_credit_mon' in df.columns:
            earliest_cr = pd.to_datetime(df['earlies_credit_mon'], errors='coerce')
            if 'issue_month' in df.columns and 'year_of_loan' in df.columns:
                issue_date = pd.to_datetime(
                    df['year_of_loan'].astype(str) + '-' + df['issue_month'].astype(str) + '-01',
                    errors='coerce')
            else:
                issue_date = pd.Timestamp('2017-01-01')
            df['credit_history_months'] = ((issue_date - earliest_cr).dt.days / 30.44).fillna(0).clip(lower=0)
            df.drop(columns=['earlies_credit_mon'], inplace=True, errors='ignore')
        else:
            df['credit_history_months'] = 0.0
        if 'scoring_low' in df.columns and 'scoring_high' in df.columns:
            df['scoring_mean'] = (df['scoring_low'] + df['scoring_high']) / 2.0
            df['scoring_range'] = df['scoring_high'] - df['scoring_low']
        if 'monthly_payment' in df.columns and 'total_loan' in df.columns:
            df['payment_loan_ratio'] = df['monthly_payment'] / (df['total_loan'] + 1e-8)
        if 'del_in_18month' in df.columns:
            df['delinq_density'] = df['del_in_18month'] / 18.0
        cat_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        for c in cat_cols:
            df[c] = df[c].astype('category').cat.codes
        target = df[target_col].values.astype(np.int64)
        df = df.drop(columns=[target_col]).fillna(0)

    else:
        for c in df.columns:
            if c != target_col:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.fillna(0)
        target = df[target_col].values.astype(np.int64)
        df = df.drop(columns=[target_col]).fillna(0)
        if 'LIMIT_BAL' in df.columns:
            limit_bal = df['LIMIT_BAL'].replace(0, 1).values.astype(np.float32)
            bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
            pay_amt_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
            for bc in bill_cols:
                if bc in df.columns:
                    df[bc + '_util'] = df[bc].values.astype(np.float32) / limit_bal
            for i in range(len(bill_cols)):
                bc = bill_cols[i]
                pc = pay_amt_cols[i]
                if bc in df.columns and pc in df.columns:
                    df[pc + '_cover'] = df[pc].values.astype(np.float32) / (np.abs(df[bc].values.astype(np.float32)) + 1.0)
            pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
            if all(c in df.columns for c in pay_cols):
                pay_matrix = df[pay_cols].values.astype(np.float32)
                df['pay_max'] = pay_matrix.max(axis=1)
                df['pay_mean'] = pay_matrix.mean(axis=1)
                df['pay_delay_cnt'] = (pay_matrix > 0).sum(axis=1)
                df['pay_trend'] = pay_matrix[:, -1] - pay_matrix[:, 0]
            if all(c in df.columns for c in bill_cols):
                bill_matrix = df[bill_cols].values.astype(np.float32)
                df['bill_mean'] = bill_matrix.mean(axis=1)
                df['bill_std'] = bill_matrix.std(axis=1)
                df['bill_trend'] = bill_matrix[:, -1] - bill_matrix[:, 0]
            if all(c in df.columns for c in pay_amt_cols):
                payamt_matrix = df[pay_amt_cols].values.astype(np.float32)
                df['payamt_mean'] = payamt_matrix.mean(axis=1)
                df['payamt_std'] = payamt_matrix.std(axis=1)
                df['zero_pay_cnt'] = (payamt_matrix == 0).sum(axis=1)
        bill_cols_full = ['BILL_AMT6','BILL_AMT5','BILL_AMT4','BILL_AMT3','BILL_AMT2','BILL_AMT1']
        pay_amt_cols_full = ['PAY_AMT6','PAY_AMT5','PAY_AMT4','PAY_AMT3','PAY_AMT2','PAY_AMT1']
        pay_stat_cols_full = ['PAY_6','PAY_5','PAY_4','PAY_3','PAY_2','PAY_0']
        eps = 1e-6
        if all(c in df.columns for c in bill_cols_full):
            bill = df[bill_cols_full].values.astype(np.float32)
            bill_diff = np.diff(bill, axis=1)
            df['bill_momentum']   = (bill_diff / (np.abs(bill[:, :-1]) + eps)).mean(axis=1)
            df['bill_roll3_mean'] = bill[:, 3:6].mean(axis=1)
            df['bill_roll3_std']  = bill[:, 3:6].std(axis=1)
            df['bill_trend_slope']= (bill[:, -1] - bill[:, 0]) / 5.0
        if all(c in df.columns for c in pay_amt_cols_full):
            pay = df[pay_amt_cols_full].values.astype(np.float32)
            pay_diff = np.diff(pay, axis=1)
            df['pay_momentum']    = (pay_diff / (np.abs(pay[:, :-1]) + eps)).mean(axis=1)
            df['pay_roll3_mean']  = pay[:, 3:6].mean(axis=1)
            df['pay_roll3_std']   = pay[:, 3:6].std(axis=1)
            df['pay_trend_slope'] = (pay[:, -1] - pay[:, 0]) / 5.0
        if all(c in df.columns for c in pay_stat_cols_full):
            stat = df[pay_stat_cols_full].values.astype(np.float32)
            df['cumulative_overdue'] = np.clip(stat, 0, None).sum(axis=1)
            df['pay_jumps']          = np.abs(np.diff(stat, axis=1)).sum(axis=1)

    if dataset_name == 'lendingclub':
        w1 = ['credit_history_months', 'total_acc', 'mo_sin_old_il_acct',
              'mo_sin_old_rev_tl_op', 'num_il_tl', 'num_rev_accts',
              'num_bc_tl', 'pub_rec', 'pub_rec_bankruptcies']
        w2 = ['delinq_2yrs', 'mths_since_last_delinq', 'mths_since_last_record',
              'mths_since_last_major_derog', 'num_accts_ever_120_pd',
              'acc_open_past_24mths', 'pct_tl_nvr_dlq', 'num_tl_90g_dpd_24m', 'mort_acc']
        w3 = ['open_il_12m', 'open_rv_12m', 'num_tl_op_past_12m',
              'inq_last_12m', 'collections_12_mths_ex_med',
              'chargeoff_within_12_mths', 'mths_since_recent_bc',
              'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq']
        w4 = ['inq_last_6mths', 'open_acc_6m', 'mths_since_rcnt_il',
              'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_inq',
              'num_tl_30dpd', 'num_tl_120dpd_2m', 'acc_now_delinq',
              'inq_acceleration', 'open_acceleration']
        w5 = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
              'revol_bal', 'revol_util', 'open_acc', 'total_rev_hi_lim',
              'tot_cur_bal', 'tot_coll_amt', 'total_bal_il', 'il_util',
              'all_util', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
              'max_bal_bc', 'num_actv_bc_tl', 'num_actv_rev_tl',
              'num_bc_sats', 'num_op_rev_tl', 'num_rev_tl_bal_gt_0',
              'num_sats', 'percent_bc_gt_75', 'tot_hi_cred_lim',
              'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
              'delinq_amnt']
    elif dataset_name == 'gmsc':
        w1 = ['n30_59', 'n30_59_flag']
        w2 = ['n60_89', 'n60_89_flag']
        w3 = ['n90_late', 'n90_late_flag']
        w4 = ['delinq_total']
        w5 = ['delinq_severity']
    elif dataset_name == 'internet':
        w1 = ['credit_history_months', 'work_year', 'house_exist', 'known_outstanding_loan']
        w2 = ['total_loan', 'interest', 'monthly_payment', 'debt_loan_ratio', 'payment_loan_ratio', 'year_of_loan',
              'issue_month', 'use']
        w3 = ['class', 'sub_class', 'scoring_low', 'scoring_high', 'scoring_mean']
        w4 = ['known_dero', 'pub_dero_bankrup', 'del_in_18month', 'delinq_dero_total']
        w5 = ['recircle_bal', 'recircle_util', 'high_recircle_util_flag', 'initial_list_status', 'app_type']
    else:
        w1 = ['PAY_6', 'BILL_AMT6', 'PAY_AMT6', 'cumulative_overdue']
        w2 = ['PAY_5', 'BILL_AMT5', 'PAY_AMT5', 'pay_jumps']
        w3 = ['PAY_4', 'PAY_3','BILL_AMT4', 'BILL_AMT3','PAY_AMT4', 'PAY_AMT3',
              'bill_roll3_mean', 'pay_roll3_mean']
        w4 = ['PAY_2','BILL_AMT2','PAY_AMT2', 'bill_momentum', 'pay_momentum']
        w5 = ['PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'bill_trend_slope', 'pay_trend_slope']

    all_windows = [w1, w2, w3, w4, w5]
    max_dim = max(len(w) for w in all_windows)

    temporal_seqs = []
    for w_cols in all_windows:
        feats = []
        for c in w_cols:
            if c in df.columns:
                feats.append(df[c].values.astype(np.float32))
            else:
                feats.append(np.zeros(len(df), dtype=np.float32))
        while len(feats) < max_dim:
            feats.append(np.zeros(len(df), dtype=np.float32))
        temporal_seqs.append(np.stack(feats, axis=1))

    temporal_data = np.stack(temporal_seqs, axis=1)

    temporal_used = set(sum(all_windows, []))
    static_cols = [c for c in df.columns if c not in temporal_used]
    if len(static_cols) == 0:
        if dataset_name == 'gmsc':
            fallback_cols = ['revolving_util', 'age', 'n30_59', 'debt_ratio', 'monthly_income',
                             'open_credit_lines', 'n90_late', 'real_estate_loans', 'n60_89', 'n_dependents']
        elif dataset_name == 'lendingclub':
            fallback_cols = ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_util']
        elif dataset_name == 'internet':
            fallback_cols = ['total_loan', 'interest', 'debt_loan_ratio', 'recircle_util', 'work_year']
        else:
            fallback_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
        static_cols = [c for c in fallback_cols if c in df.columns]
    if len(static_cols) == 0:
        raise ValueError("静态特征仍为空，请检查字段映射。")

    static_data = df[static_cols].values.astype(np.float32)

    def pick(cols):
        return [c for c in cols if c in static_cols]

    if dataset_name == 'lendingclub':
        graph_groups_def = {
            'demographic': ['addr_state', 'zip_code', 'home_ownership'],
            'credit': ['grade', 'sub_grade'],
            'financial': ['annual_inc', 'dti', 'revol_util'],
            'behavior': ['loan_amnt', 'term', 'purpose', 'int_rate'],
        }
    elif dataset_name == 'gmsc':
        graph_groups_def = {
            'demographic': pick(['age', 'n_dependents', 'income_per_dep']),
            'credit': pick(['open_credit_lines', 'real_estate_loans', 'credit_mix']),
            'financial': pick(['monthly_income', 'debt_ratio', 'debt_amount_est']),
            'behavior': pick(['revolving_util']),
        }
    elif dataset_name == 'internet':
        def pick_static(cols):
            return [c for c in cols if c in static_cols]
        graph_groups_def = {
            'behavior': pick_static(['f0', 'f1', 'f2', 'f3', 'f4', 'f5']),
            'credit': pick_static(['house_ownership', 'house_loan_status', 'marriage', 'offsprings']),
            'demographic': pick_static(['post_code', 'region']),
            'financial': pick_static(['work_type', 'employment_type', 'industry']),
        }
    else:
        graph_groups_def = {
            'demographic': ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE'],
            'credit': ['LIMIT_BAL'],
            'financial': ['bill_mean', 'bill_std', 'payamt_mean', 'payamt_std'],
            'behavior': ['pay_max', 'pay_mean', 'pay_delay_cnt', 'pay_trend'],
        }

    graph_group_indices = []
    for _, gcols in graph_groups_def.items():
        idxs = [static_cols.index(c) for c in gcols if c in static_cols]
        graph_group_indices.append(idxs if len(idxs) > 0 else [])

    N, T, D = temporal_data.shape
    scaler_t = StandardScaler()
    temporal_data = scaler_t.fit_transform(temporal_data.reshape(N, -1)).reshape(N, T, D)

    scaler_s = StandardScaler()
    static_data = scaler_s.fit_transform(static_data)

    corr = pd.DataFrame(static_data, columns=static_cols).corr()
    drop_set = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.6:
                drop_set.add(corr.columns[i])

    keep_idx = [i for i, c in enumerate(static_cols) if c not in drop_set]
    if len(keep_idx) == 0:
        keep_idx = list(range(len(static_cols)))

    static_cols_kept = [static_cols[i] for i in keep_idx]
    static_data = static_data[:, keep_idx]

    new_groups = []
    for old in graph_group_indices:
        tmp = []
        for oi in old:
            cname = static_cols[oi]
            if cname in static_cols_kept:
                tmp.append(static_cols_kept.index(cname))
        new_groups.append(tmp)
    graph_group_indices = new_groups

    group_dims = []
    for g in graph_group_indices:
        group_dims.append(len(g) if len(g) > 0 else static_data.shape[1])

    return temporal_data, static_data, target, graph_group_indices, group_dims, T, D


temporal_data, static_data, target, graph_group_indices, group_dims, T, temporal_input_dim = load_dataset(DATASET)

idx_all = np.arange(len(target))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx_all, target, test_size=0.2, random_state=SEED, stratify=target
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

Xt_train = torch.tensor(temporal_data[idx_train], dtype=torch.float32)
Xt_val   = torch.tensor(temporal_data[idx_val],   dtype=torch.float32)
Xt_test  = torch.tensor(temporal_data[idx_test],  dtype=torch.float32)
Xs_train = torch.tensor(static_data[idx_train], dtype=torch.float32)
Xs_val   = torch.tensor(static_data[idx_val],   dtype=torch.float32)
Xs_test  = torch.tensor(static_data[idx_test],  dtype=torch.float32)
yt_train = torch.tensor(y_train, dtype=torch.float32)
yt_val   = torch.tensor(y_val,   dtype=torch.float32)
yt_test  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xt_train, Xs_train, yt_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xt_val,   Xs_val,   yt_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xt_test,  Xs_test,  yt_test),  batch_size=BATCH_SIZE, shuffle=False)

cls_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
cls_weights = torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE)
alpha_neg, alpha_pos = cls_weights[0].item(), cls_weights[1].item()
total = alpha_neg + alpha_pos
FOCAL_ALPHA_NEG = alpha_neg / total
FOCAL_ALPHA_POS = alpha_pos / total

class ChangePointAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.change_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads)
        )
        self.temp = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        x_diff = torch.zeros_like(x)
        x_diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        change = self.change_detector(x_diff).permute(0, 2, 1)
        attn = attn + self.temp * change.unsqueeze(2).expand_as(attn)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, seq_len=5, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': ChangePointAwareAttention(d_model, n_heads, dropout),
                'n1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
                ),
                'n2': nn.LayerNorm(d_model)
            }))
        self.pool = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.size(1), :]
        for blk in self.layers:
            h = blk['n1'](h + blk['attn'](h))
            h = blk['n2'](h + blk['ffn'](h))
        w = torch.softmax(self.pool(h).squeeze(-1), dim=-1)
        out = (w.unsqueeze(-1) * h).sum(dim=1)
        return out


class DynamicGraphConstructor(nn.Module):
    def __init__(self, static_dim, group_dims, n_edge_types=4, k=8):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.k = k
        self.projs = nn.ModuleList()
        for t in range(n_edge_types):
            in_dim = group_dims[t] if t < len(group_dims) else static_dim
            in_dim = max(in_dim, 1)
            h = max(in_dim * 2, 4)
            z = max(in_dim, 4)
            self.projs.append(nn.Sequential(
                nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, z)
            ))
        self.thresholds = nn.Parameter(torch.ones(n_edge_types) * 0.2)

    def forward(self, x_static, group_indices):
        B = x_static.size(0)
        all_src, all_dst, all_typ, all_w = [], [], [], []

        for t in range(self.n_edge_types):
            if t < len(group_indices) and len(group_indices[t]) > 0:
                xs = x_static[:, group_indices[t]]
            else:
                xs = x_static

            z = F.normalize(self.projs[t](xs), p=2, dim=-1)
            sim = z @ z.T
            sim.fill_diagonal_(-1e9)

            k = min(self.k, B - 1)
            if k <= 0:
                continue

            vals, idx = torch.topk(sim, k=k, dim=-1)
            thr = torch.sigmoid(self.thresholds[t])
            mask = vals > thr

            src = torch.arange(B, device=x_static.device).unsqueeze(1).expand_as(idx)
            if mask.any():
                all_src.append(src[mask])
                all_dst.append(idx[mask])
                all_typ.append(torch.full_like(src[mask], t))
                all_w.append(vals[mask])

        if len(all_src) == 0:
            z0 = torch.zeros(0, dtype=torch.long, device=x_static.device)
            return z0, z0, z0, torch.zeros(0, device=x_static.device)

        return torch.cat(all_src), torch.cat(all_dst), torch.cat(all_typ), torch.cat(all_w)


class GraphEncoder(nn.Module):
    def __init__(self, static_dim, group_dims, d_model=64, n_heads=4, n_edge_types=4, k=8, dropout=0.3):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.builder = DynamicGraphConstructor(static_dim, group_dims, n_edge_types=n_edge_types, k=k)
        self.node_embed = nn.Linear(static_dim, d_model)

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.edge_bias = nn.Parameter(torch.randn(n_edge_types, n_heads) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)

        self.n1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x_static, group_indices):
        B = x_static.size(0)
        x = self.node_embed(x_static)

        src, dst, etype, ew = self.builder(x_static, group_indices)
        if src.numel() == 0:
            x = self.n2(x + self.ffn(self.n1(x)))
            return x

        Q = self.WQ(x).view(B, self.n_heads, self.d_k)
        K = self.WK(x).view(B, self.n_heads, self.d_k)
        V = self.WV(x).view(B, self.n_heads, self.d_k)

        q = Q[src]
        k = K[dst]
        v = V[dst]

        score = (q * k).sum(-1) / math.sqrt(self.d_k)
        score = score + self.edge_bias[etype] + ew.unsqueeze(-1)

        max_per_node = torch.full((B, self.n_heads), -1e9, device=x.device)
        max_per_node.scatter_reduce_(0, src.unsqueeze(-1).expand_as(score), score, reduce='amax')
        score_exp = torch.exp(score - max_per_node[src])

        sum_per_node = torch.zeros(B, self.n_heads, device=x.device)
        sum_per_node.scatter_add_(0, src.unsqueeze(-1).expand_as(score_exp), score_exp)
        alpha = score_exp / (sum_per_node[src] + 1e-8)

        msg = alpha.unsqueeze(-1) * v
        out = torch.zeros(B, self.n_heads, self.d_k, device=x.device)
        out.scatter_add_(0, src.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        out = out.reshape(B, self.d_model)

        x = self.n1(x + self.out_proj(out))
        x = self.n2(x + self.ffn(x))
        return x


class TimeSpaceEnhancedTransformer(nn.Module):
    def __init__(self, temporal_input_dim, static_dim, group_dims, seq_len=5, d_model=64, n_heads=4):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim, seq_len=seq_len,
            d_model=d_model, n_heads=n_heads, n_layers=2
        )
        self.graph_encoder = GraphEncoder(
            static_dim=static_dim, group_dims=group_dims,
            d_model=d_model, n_heads=n_heads, n_edge_types=4, k=8
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_temp, x_static, group_indices, return_weights=False):
        h_t = self.temporal_encoder(x_temp)
        h_s = self.graph_encoder(x_static, group_indices)

        gate_logits = self.gate(torch.cat([h_t, h_s], dim=-1))
        gate_w = torch.softmax(gate_logits, dim=-1)

        w_t = gate_w[:, 0].unsqueeze(-1)
        w_s = gate_w[:, 1].unsqueeze(-1)

        h = w_t * h_t + w_s * h_s
        y = self.cls(h).squeeze(-1)

        if return_weights:
            return y, gate_w
        return y


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1 - self.eps)
        pt_pos = pred
        pt_neg = 1 - pred

        loss_pos = -self.alpha_pos * ((1 - pt_pos) ** self.gamma) * torch.log(pt_pos) * target
        loss_neg = -self.alpha_neg * ((1 - pt_neg) ** self.gamma) * torch.log(pt_neg) * (1 - target)
        return (loss_pos + loss_neg).mean()


def gate_entropy_regularization(gate_w, eps=1e-8):
    ent = -(gate_w * torch.log(gate_w + eps)).sum(dim=-1).mean()
    return -ent


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if (self.best_score is None) or (val_score > self.best_score + self.min_delta):
            self.best_score = val_score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


model = TimeSpaceEnhancedTransformer(
    temporal_input_dim=temporal_input_dim,
    static_dim=static_data.shape[1],
    group_dims=group_dims,
    seq_len=T,
    d_model=32,
    n_heads=2
).to(DEVICE)

criterion = WeightedFocalLoss(alpha_pos=FOCAL_ALPHA_POS, alpha_neg=FOCAL_ALPHA_NEG, gamma=FOCAL_GAMMA)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def warmup_lambda(epoch, warmup_epochs=5):
    if epoch < warmup_epochs: return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: warmup_lambda(e, 5))
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 5, eta_min=1e-6)
early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
LABEL_SMOOTH = 0.05

train_loss_curve = []
val_loss_curve = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb_t, xb_s, yb in train_loader:
        xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred, gate_w = model(xb_t, xb_s, graph_group_indices, return_weights=True)
        yb_smooth = yb * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
        loss_cls = criterion(pred, yb_smooth)
        loss_gate = gate_entropy_regularization(gate_w)
        loss = loss_cls + GATE_ENTROPY_LAMBDA * loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    if epoch <= 5: warmup_scheduler.step()
    else: cosine_scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb_t, xb_s, yb in val_loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            prob, gate_w = model(xb_t, xb_s, graph_group_indices, return_weights=True)
            yb_smooth = yb * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            loss_c = criterion(prob, yb_smooth)
            loss_g = gate_entropy_regularization(gate_w)
            val_losses.append((loss_c + GATE_ENTROPY_LAMBDA * loss_g).item())

    val_loss = np.mean(val_losses)
    train_loss_curve.append(np.mean(train_losses))
    val_loss_curve.append(val_loss)

    early_stopper(-val_loss, model)
    if early_stopper.early_stop:
        break

if early_stopper.best_state is not None:
    model.load_state_dict(early_stopper.best_state)

import copy
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

COLOR_DEEP_CRIMSON = '#B2182B'
COLOR_BRICK_RED    = '#D6604D'
COLOR_PEACH_ORANGE = '#F4A582'
COLOR_LIGHT_APRICOT= '#FDDBC7'
COLOR_MIST_BLUE    = '#D1E5F0'
COLOR_SKY_BLUE     = '#92C5DE'
COLOR_LAKE_BLUE    = '#4393C3'
COLOR_DEEP_SEA     = '#2166AC'

DIAG_SAVE_DIR = os.path.join(SCRIPT_DIR, 'results', 'RQ3_3')
os.makedirs(DIAG_SAVE_DIR, exist_ok=True)


class FocalLossAdaptive(nn.Module):
    def __init__(self, pos_rate, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha_pos = 1 - pos_rate
        self.alpha_neg = pos_rate
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1 - self.eps)
        pt_pos = pred
        pt_neg = 1 - pred
        loss_pos = -self.alpha_pos * ((1 - pt_pos) ** self.gamma) * torch.log(pt_pos) * target
        loss_neg = -self.alpha_neg * ((1 - pt_neg) ** self.gamma) * torch.log(pt_neg) * (1 - target)
        return (loss_pos + loss_neg).mean()


class FocalLossFixed(nn.Module):
    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1 - self.eps)
        pt_pos = pred
        pt_neg = 1 - pred
        loss_pos = -self.alpha_pos * ((1 - pt_pos) ** self.gamma) * torch.log(pt_pos) * target
        loss_neg = -self.alpha_neg * ((1 - pt_neg) ** self.gamma) * torch.log(pt_neg) * (1 - target)
        return (loss_pos + loss_neg).mean()


class BCELossWrapper(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1 - self.eps)
        loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        return loss.mean()


def _diag_train_model(loss_fn, train_loader, graph_group_indices,
                      temporal_input_dim, static_dim, group_dims, T,
                      epochs=30, lr=5e-4, weight_decay=1e-3,
                      use_label_smoothing=False, epsilon=0.05,
                      record_history=False, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mdl = TimeSpaceEnhancedTransformer(
        temporal_input_dim=temporal_input_dim,
        static_dim=static_dim,
        group_dims=group_dims,
        seq_len=T,
        d_model=32,
        n_heads=2
    ).to(DEVICE)

    opt = optim.AdamW(mdl.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    for ep in range(epochs):
        mdl.train()
        ep_losses = []
        all_preds_tr, all_labels_tr = [], []

        for xb_t, xb_s, yb in train_loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = mdl(xb_t, xb_s, graph_group_indices)

            if use_label_smoothing:
                yb_used = yb * (1 - epsilon) + 0.5 * epsilon
            else:
                yb_used = yb

            loss = loss_fn(pred, yb_used)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
            opt.step()

            ep_losses.append(loss.item())
            all_preds_tr.extend(pred.detach().cpu().numpy())
            all_labels_tr.extend(yb.cpu().numpy())

        if record_history:
            history['train_loss'].append(np.mean(ep_losses))
            try:
                history['train_auc'].append(roc_auc_score(all_labels_tr, all_preds_tr))
            except:
                history['train_auc'].append(0.5)

    return mdl, history


def _diag_evaluate(mdl, loader, graph_group_indices):
    mdl.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb_t, xb_s, yb in loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            pred = mdl(xb_t, xb_s, graph_group_indices)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return roc_auc_score(all_labels, all_preds)


def _diag_evaluate_loss(mdl, loader, loss_fn, graph_group_indices,
                        use_label_smoothing=False, epsilon=0.05):
    mdl.eval()
    losses = []
    with torch.no_grad():
        for xb_t, xb_s, yb in loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            pred = mdl(xb_t, xb_s, graph_group_indices)
            if use_label_smoothing:
                yb_used = yb * (1 - epsilon) + 0.5 * epsilon
            else:
                yb_used = yb
            loss = loss_fn(pred, yb_used)
            losses.append(loss.item())
    return np.mean(losses)


def exp_gradient_balance(train_loader, val_loader, graph_group_indices,
                         temporal_input_dim, static_dim, group_dims, T,
                         pos_rate, dataset_name, seed=42):

    loss_configs = {
        'BCE': BCELossWrapper(),
        'Focal_fixed (a=0.75)': FocalLossFixed(alpha_pos=0.75, alpha_neg=0.25, gamma=FOCAL_GAMMA),
        'Focal_adaptive': FocalLossAdaptive(pos_rate=pos_rate, gamma=FOCAL_GAMMA),
    }

    n_trials = 3
    gradient_results = {}

    for name, loss_fn in loss_configs.items():
        pos_grads_all, neg_grads_all = [], []

        for trial in range(n_trials):
            mdl, _ = _diag_train_model(
                loss_fn, train_loader, graph_group_indices,
                temporal_input_dim, static_dim, group_dims, T,
                epochs=15, lr=LR, weight_decay=WEIGHT_DECAY, seed=seed + trial
            )
            mdl.eval()

            pos_grad_norms, neg_grad_norms = [], []

            for xb_t, xb_s, yb in train_loader:
                xb_t = xb_t.to(DEVICE).requires_grad_(True)
                xb_s = xb_s.to(DEVICE).requires_grad_(True)
                yb = yb.to(DEVICE)

                pred = mdl(xb_t, xb_s, graph_group_indices)
                pred_clamped = pred.clamp(1e-8, 1 - 1e-8)
                pt_pos = pred_clamped
                pt_neg = 1 - pred_clamped
                if 'adaptive' in name.lower():
                    a_p, a_n = 1 - pos_rate, pos_rate
                elif 'fixed' in name.lower():
                    a_p, a_n = 0.75, 0.25
                else:
                    a_p, a_n = 1.0, 1.0

                l_pos = -a_p * ((1 - pt_pos) ** FOCAL_GAMMA) * torch.log(pt_pos) * yb
                l_neg = -a_n * ((1 - pt_neg) ** FOCAL_GAMMA) * torch.log(pt_neg) * (1 - yb)
                per_sample_loss = l_pos + l_neg

                for i in range(len(yb)):
                    mdl.zero_grad()
                    if xb_t.grad is not None:
                        xb_t.grad.zero_()
                    per_sample_loss[i].backward(retain_graph=True)
                    grad_norm = xb_t.grad[i].norm().item() if xb_t.grad is not None else 0.0
                    if yb[i].item() == 1:
                        pos_grad_norms.append(grad_norm)
                    else:
                        neg_grad_norms.append(grad_norm)

                break

            pos_grads_all.append(np.mean(pos_grad_norms) if len(pos_grad_norms) > 0 else 0)
            neg_grads_all.append(np.mean(neg_grad_norms) if len(neg_grad_norms) > 0 else 0)

        mean_pos = np.mean(pos_grads_all)
        mean_neg = np.mean(neg_grads_all)
        ratio = mean_pos / (mean_neg + 1e-10)
        gradient_results[name] = {
            'pos_grad': mean_pos,
            'neg_grad': mean_neg,
            'ratio': ratio,
            'deviation': abs(ratio - 1.0)
        }
        print(f"    {name:30s} -> Pos Grad: {mean_pos:.6f}, Neg Grad: {mean_neg:.6f}, "
              f"Ratio: {ratio:.3f}, Deviation: {abs(ratio - 1.0):.3f}")

    print(f"\n    [Result] Ideal gradient ratio = 1.0 (balanced)")
    adaptive_dev = gradient_results['Focal_adaptive']['deviation']
    fixed_dev = gradient_results['Focal_fixed (a=0.75)']['deviation']
    print(f"    [Result] Adaptive deviation: {adaptive_dev:.3f}, Fixed deviation: {fixed_dev:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    names = list(gradient_results.keys())
    pos_vals = [gradient_results[n]['pos_grad'] for n in names]
    neg_vals = [gradient_results[n]['neg_grad'] for n in names]

    x_pos = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, pos_vals, width, label='Positive Samples',
                   color=COLOR_DEEP_CRIMSON, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x_pos + width / 2, neg_vals, width, label='Negative Samples',
                   color=COLOR_DEEP_SEA, edgecolor='white', linewidth=0.8)

    ax.set_xlabel('Loss Function', fontsize=12)
    ax.set_ylabel('Mean Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Balance Analysis - {dataset_name}', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for i, n in enumerate(names):
        r = gradient_results[n]['ratio']
        ax.annotate(f'Ratio={r:.2f}', xy=(x_pos[i], max(pos_vals[i], neg_vals[i])),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', color=COLOR_BRICK_RED)

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_SAVE_DIR, f'exp332_gradient_balance_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return gradient_results


def exp_label_smoothing_robustness(Xt_train, Xs_train, yt_train,
                                   val_loader, graph_group_indices,
                                   temporal_input_dim, static_dim, group_dims, T,
                                   pos_rate, dataset_name, seed=42):

    noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    n_trials = 3
    results_no_ls = {}
    results_with_ls = {}

    for nr in noise_rates:
        aucs_no, aucs_ls = [], []

        for trial in range(n_trials):
            rng = np.random.RandomState(seed + trial)
            y_noisy = yt_train.clone()

            if nr > 0:
                n_flip = int(len(y_noisy) * nr)
                flip_idx = rng.choice(len(y_noisy), n_flip, replace=False)
                y_noisy[flip_idx] = 1.0 - y_noisy[flip_idx]

            noisy_loader = DataLoader(
                TensorDataset(Xt_train, Xs_train, y_noisy),
                batch_size=BATCH_SIZE, shuffle=True, drop_last=True
            )

            loss_fn = FocalLossAdaptive(pos_rate=pos_rate, gamma=FOCAL_GAMMA)

            mdl_no, _ = _diag_train_model(
                loss_fn, noisy_loader, graph_group_indices,
                temporal_input_dim, static_dim, group_dims, T,
                epochs=20, lr=LR, weight_decay=WEIGHT_DECAY,
                use_label_smoothing=False, seed=seed + trial
            )
            aucs_no.append(_diag_evaluate(mdl_no, val_loader, graph_group_indices))

            mdl_ls, _ = _diag_train_model(
                loss_fn, noisy_loader, graph_group_indices,
                temporal_input_dim, static_dim, group_dims, T,
                epochs=20, lr=LR, weight_decay=WEIGHT_DECAY,
                use_label_smoothing=True, epsilon=0.05, seed=seed + trial
            )
            aucs_ls.append(_diag_evaluate(mdl_ls, val_loader, graph_group_indices))

        results_no_ls[nr] = {'mean': np.mean(aucs_no), 'std': np.std(aucs_no)}
        results_with_ls[nr] = {'mean': np.mean(aucs_ls), 'std': np.std(aucs_ls)}

        print(f"    Noise={nr:.0%}: No LS={np.mean(aucs_no):.4f}+/-{np.std(aucs_no):.4f}, "
              f"With LS={np.mean(aucs_ls):.4f}+/-{np.std(aucs_ls):.4f}")

    improve_20 = results_with_ls[0.20]['mean'] - results_no_ls[0.20]['mean']
    print(f"\n    [Result] At 20% noise, LS improvement: {improve_20:+.4f} AUC")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    nrs = noise_rates
    means_no = [results_no_ls[n]['mean'] for n in nrs]
    stds_no = [results_no_ls[n]['std'] for n in nrs]
    means_ls = [results_with_ls[n]['mean'] for n in nrs]
    stds_ls = [results_with_ls[n]['std'] for n in nrs]

    ax.errorbar(nrs, means_no, yerr=stds_no, marker='s', color=COLOR_BRICK_RED,
                linewidth=2, markersize=8, capsize=4, label='Without Label Smoothing')
    ax.errorbar(nrs, means_ls, yerr=stds_ls, marker='o', color=COLOR_DEEP_SEA,
                linewidth=2, markersize=8, capsize=4, label='With Label Smoothing (eps=0.05)')

    ax.fill_between(nrs, [m - s for m, s in zip(means_no, stds_no)],
                    [m + s for m, s in zip(means_no, stds_no)],
                    alpha=0.15, color=COLOR_PEACH_ORANGE)
    ax.fill_between(nrs, [m - s for m, s in zip(means_ls, stds_ls)],
                    [m + s for m, s in zip(means_ls, stds_ls)],
                    alpha=0.15, color=COLOR_MIST_BLUE)

    ax.set_xlabel('Label Noise Rate', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title(f'Label Smoothing Robustness Under Noise - {dataset_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nrs)
    ax.set_xticklabels([f'{int(n*100)}%' for n in nrs])
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_SAVE_DIR, f'exp333_label_smoothing_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {'no_ls': results_no_ls, 'with_ls': results_with_ls}


def exp_overfitting_gap(train_loader, val_loader, graph_group_indices,
                        temporal_input_dim, static_dim, group_dims, T,
                        pos_rate, dataset_name, seed=42):

    epochs = 40
    loss_fn = FocalLossAdaptive(pos_rate=pos_rate, gamma=FOCAL_GAMMA)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mdl_no = TimeSpaceEnhancedTransformer(
        temporal_input_dim=temporal_input_dim,
        static_dim=static_dim,
        group_dims=group_dims,
        seq_len=T, d_model=32, n_heads=2
    ).to(DEVICE)
    opt_no = optim.AdamW(mdl_no.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist_no = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    for ep in range(epochs):
        mdl_no.train()
        ep_loss, preds_tr, labels_tr = [], [], []
        for xb_t, xb_s, yb in train_loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            opt_no.zero_grad()
            pred = mdl_no(xb_t, xb_s, graph_group_indices)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl_no.parameters(), max_norm=1.0)
            opt_no.step()
            ep_loss.append(loss.item())
            preds_tr.extend(pred.detach().cpu().numpy())
            labels_tr.extend(yb.cpu().numpy())

        hist_no['train_loss'].append(np.mean(ep_loss))
        try:
            hist_no['train_auc'].append(roc_auc_score(labels_tr, preds_tr))
        except:
            hist_no['train_auc'].append(0.5)

        mdl_no.eval()
        vl, preds_v, labels_v = [], [], []
        with torch.no_grad():
            for xb_t, xb_s, yb in val_loader:
                xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
                pred = mdl_no(xb_t, xb_s, graph_group_indices)
                vl.append(loss_fn(pred, yb).item())
                preds_v.extend(pred.cpu().numpy())
                labels_v.extend(yb.cpu().numpy())
        hist_no['val_loss'].append(np.mean(vl))
        try:
            hist_no['val_auc'].append(roc_auc_score(labels_v, preds_v))
        except:
            hist_no['val_auc'].append(0.5)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mdl_ls = TimeSpaceEnhancedTransformer(
        temporal_input_dim=temporal_input_dim,
        static_dim=static_dim,
        group_dims=group_dims,
        seq_len=T, d_model=32, n_heads=2
    ).to(DEVICE)
    opt_ls = optim.AdamW(mdl_ls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist_ls = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    eps_smooth = 0.05

    for ep in range(epochs):
        mdl_ls.train()
        ep_loss, preds_tr, labels_tr = [], [], []
        for xb_t, xb_s, yb in train_loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            opt_ls.zero_grad()
            pred = mdl_ls(xb_t, xb_s, graph_group_indices)
            yb_smooth = yb * (1 - eps_smooth) + 0.5 * eps_smooth
            loss = loss_fn(pred, yb_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl_ls.parameters(), max_norm=1.0)
            opt_ls.step()
            ep_loss.append(loss.item())
            preds_tr.extend(pred.detach().cpu().numpy())
            labels_tr.extend(yb.cpu().numpy())

        hist_ls['train_loss'].append(np.mean(ep_loss))
        try:
            hist_ls['train_auc'].append(roc_auc_score(labels_tr, preds_tr))
        except:
            hist_ls['train_auc'].append(0.5)

        mdl_ls.eval()
        vl, preds_v, labels_v = [], [], []
        with torch.no_grad():
            for xb_t, xb_s, yb in val_loader:
                xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
                pred = mdl_ls(xb_t, xb_s, graph_group_indices)
                yb_smooth = yb * (1 - eps_smooth) + 0.5 * eps_smooth
                vl.append(loss_fn(pred, yb_smooth).item())
                preds_v.extend(pred.cpu().numpy())
                labels_v.extend(yb.cpu().numpy())
        hist_ls['val_loss'].append(np.mean(vl))
        try:
            hist_ls['val_auc'].append(roc_auc_score(labels_v, preds_v))
        except:
            hist_ls['val_auc'].append(0.5)

    gap_no = hist_no['train_auc'][-1] - hist_no['val_auc'][-1]
    gap_ls = hist_ls['train_auc'][-1] - hist_ls['val_auc'][-1]
    print(f"    Without LS: Train AUC={hist_no['train_auc'][-1]:.4f}, "
          f"Val AUC={hist_no['val_auc'][-1]:.4f}, Gap={gap_no:.4f}")
    print(f"    With LS:    Train AUC={hist_ls['train_auc'][-1]:.4f}, "
          f"Val AUC={hist_ls['val_auc'][-1]:.4f}, Gap={gap_ls:.4f}")
    print(f"\n    [Result] Overfitting gap reduction: {gap_no - gap_ls:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ep_range = range(1, epochs + 1)
    ax.plot(ep_range, hist_no['train_loss'], '-', color=COLOR_PEACH_ORANGE, linewidth=2, label='Train Loss (No LS)')
    ax.plot(ep_range, hist_no['val_loss'], '--', color=COLOR_BRICK_RED, linewidth=2, label='Val Loss (No LS)')
    ax.plot(ep_range, hist_ls['train_loss'], '-', color=COLOR_SKY_BLUE, linewidth=2, label='Train Loss (With LS)')
    ax.plot(ep_range, hist_ls['val_loss'], '--', color=COLOR_DEEP_SEA, linewidth=2, label='Val Loss (With LS)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training/Validation Loss - {dataset_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ep_range, hist_no['train_auc'], '-', color=COLOR_PEACH_ORANGE, linewidth=2, label='Train AUC (No LS)')
    ax.plot(ep_range, hist_no['val_auc'], '--', color=COLOR_BRICK_RED, linewidth=2, label='Val AUC (No LS)')
    ax.plot(ep_range, hist_ls['train_auc'], '-', color=COLOR_SKY_BLUE, linewidth=2, label='Train AUC (With LS)')
    ax.plot(ep_range, hist_ls['val_auc'], '--', color=COLOR_DEEP_SEA, linewidth=2, label='Val AUC (With LS)')

    ax.annotate(f'Gap={gap_no:.3f}', xy=(epochs, hist_no['val_auc'][-1]),
                xytext=(epochs - 8, hist_no['val_auc'][-1] - 0.03),
                fontsize=9, color=COLOR_DEEP_CRIMSON, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_DEEP_CRIMSON))
    ax.annotate(f'Gap={gap_ls:.3f}', xy=(epochs, hist_ls['val_auc'][-1]),
                xytext=(epochs - 8, hist_ls['val_auc'][-1] + 0.02),
                fontsize=9, color=COLOR_DEEP_SEA, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_DEEP_SEA))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Overfitting Gap Analysis - {dataset_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_SAVE_DIR, f'exp334_overfitting_gap_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'no_ls': hist_no, 'with_ls': hist_ls,
        'gap_no_ls': gap_no, 'gap_with_ls': gap_ls
    }


def plot_combined_summary(all_results, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if 'alpha' in all_results:
        ax = axes[0, 0]
        alpha_res = all_results['alpha']
        alphas_plot = [k for k in alpha_res.keys() if k != 'adaptive']
        aucs_plot = [alpha_res[a] for a in alphas_plot]
        adaptive_auc = alpha_res.get('adaptive', 0)

        ax.plot(alphas_plot, aucs_plot, 'o-', color=COLOR_DEEP_SEA, linewidth=2, markersize=6)
        ax.axhline(y=adaptive_auc, color=COLOR_DEEP_CRIMSON, linestyle='--', linewidth=2.5,
                   label=f'Adaptive (AUC={adaptive_auc:.4f})')
        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel('Val AUC', fontsize=11)
        ax.set_title('(a) Alpha Sensitivity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].set_visible(False)

    ax = axes[0, 1]
    grad_res = all_results['gradient']
    names = list(grad_res.keys())
    ratios = [grad_res[n]['ratio'] for n in names]
    colors_bar = [COLOR_PEACH_ORANGE, COLOR_LAKE_BLUE, COLOR_DEEP_CRIMSON]

    bars = ax.bar(range(len(names)), ratios, color=colors_bar[:len(names)],
                  edgecolor='white', linewidth=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Ideal (1.0)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel('Gradient Ratio (Pos/Neg)', fontsize=11)
    ax.set_title('(b) Gradient Balance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9, fontweight='bold')

    ax = axes[1, 0]
    ls_res = all_results['label_smoothing']
    nrs = sorted(ls_res['no_ls'].keys())
    means_no = [ls_res['no_ls'][n]['mean'] for n in nrs]
    means_ls = [ls_res['with_ls'][n]['mean'] for n in nrs]
    stds_no = [ls_res['no_ls'][n]['std'] for n in nrs]
    stds_ls = [ls_res['with_ls'][n]['std'] for n in nrs]

    ax.errorbar(nrs, means_no, yerr=stds_no, marker='s', color=COLOR_BRICK_RED,
                linewidth=2, capsize=4, label='Without LS')
    ax.errorbar(nrs, means_ls, yerr=stds_ls, marker='o', color=COLOR_DEEP_SEA,
                linewidth=2, capsize=4, label='With LS')
    ax.set_xlabel('Noise Rate', fontsize=11)
    ax.set_ylabel('Val AUC', fontsize=11)
    ax.set_title('(c) Noise Robustness', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nrs)
    ax.set_xticklabels([f'{int(n*100)}%' for n in nrs])

    ax = axes[1, 1]
    gap_res = all_results['overfitting']
    categories = ['Without LS', 'With LS']
    gaps = [gap_res['gap_no_ls'], gap_res['gap_with_ls']]
    colors_gap = [COLOR_BRICK_RED, COLOR_DEEP_SEA]

    bars = ax.bar(categories, gaps, color=colors_gap, edgecolor='white', linewidth=0.8, width=0.5)
    ax.set_ylabel('Overfitting Gap (Train AUC - Val AUC)', fontsize=11)
    ax.set_title('(d) Overfitting Gap', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, g in enumerate(gaps):
        ax.text(i, g + 0.002, f'{g:.4f}', ha='center', fontsize=10, fontweight='bold')

    if 'alpha' in all_results:
        plt.suptitle(f'RQ3.3 Diagnostic Analysis Summary - {dataset_name}',
                     fontsize=14, fontweight='bold', y=1.01)
    else:
        plt.suptitle(f'RQ3.3 Diagnostic Analysis Summary - {dataset_name} (3 experiments)',
                     fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_SAVE_DIR, f'RQ33_summary_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def export_results_csv(all_datasets_results):
    if any('alpha' in res for res in all_datasets_results.values()):
        rows = []
        for ds, res in all_datasets_results.items():
            if 'alpha' in res:
                alpha_res = res['alpha']
                for k, v in alpha_res.items():
                    rows.append({'Dataset': ds, 'Alpha': k, 'Val_AUC': v})
        if rows:
            df_alpha = pd.DataFrame(rows)
            df_alpha.to_csv(os.path.join(DIAG_SAVE_DIR, 'table_alpha_sensitivity.csv'), index=False)

    rows = []
    for ds, res in all_datasets_results.items():
        grad_res = res['gradient']
        for name, vals in grad_res.items():
            rows.append({
                'Dataset': ds, 'Loss_Function': name,
                'Pos_Grad': vals['pos_grad'], 'Neg_Grad': vals['neg_grad'],
                'Ratio': vals['ratio'], 'Deviation': vals['deviation']
            })
    df_grad = pd.DataFrame(rows)
    df_grad.to_csv(os.path.join(DIAG_SAVE_DIR, 'table_gradient_balance.csv'), index=False)

    rows = []
    for ds, res in all_datasets_results.items():
        ls_res = res['label_smoothing']
        for nr in sorted(ls_res['no_ls'].keys()):
            rows.append({
                'Dataset': ds, 'Noise_Rate': nr,
                'AUC_No_LS_Mean': ls_res['no_ls'][nr]['mean'],
                'AUC_No_LS_Std': ls_res['no_ls'][nr]['std'],
                'AUC_With_LS_Mean': ls_res['with_ls'][nr]['mean'],
                'AUC_With_LS_Std': ls_res['with_ls'][nr]['std'],
            })
    df_ls = pd.DataFrame(rows)
    df_ls.to_csv(os.path.join(DIAG_SAVE_DIR, 'table_label_smoothing.csv'), index=False)

    rows = []
    for ds, res in all_datasets_results.items():
        gap_res = res['overfitting']
        rows.append({
            'Dataset': ds,
            'Gap_No_LS': gap_res['gap_no_ls'],
            'Gap_With_LS': gap_res['gap_with_ls'],
            'Gap_Reduction': gap_res['gap_no_ls'] - gap_res['gap_with_ls']
        })
    df_gap = pd.DataFrame(rows)
    df_gap.to_csv(os.path.join(DIAG_SAVE_DIR, 'table_overfitting_gap.csv'), index=False)

    print(f"\n  [Export] All tables saved to {DIAG_SAVE_DIR}")


def run_rq33_diagnostic(dataset_name, train_loader, val_loader,
                        Xt_train, Xs_train, yt_train,
                        graph_group_indices, temporal_input_dim,
                        static_dim, group_dims, T, pos_rate, seed=42):

    res_gradient = exp_gradient_balance(
        train_loader, val_loader, graph_group_indices,
        temporal_input_dim, static_dim, group_dims, T,
        pos_rate, dataset_name, seed
    )

    res_ls = exp_label_smoothing_robustness(
        Xt_train, Xs_train, yt_train,
        val_loader, graph_group_indices,
        temporal_input_dim, static_dim, group_dims, T,
        pos_rate, dataset_name, seed
    )

    res_gap = exp_overfitting_gap(
        train_loader, val_loader, graph_group_indices,
        temporal_input_dim, static_dim, group_dims, T,
        pos_rate, dataset_name, seed
    )

    plot_combined_summary(
        {'gradient': res_gradient,
         'label_smoothing': res_ls, 'overfitting': res_gap},
        dataset_name
    )

    return {
        'gradient': res_gradient,
        'label_smoothing': res_ls,
        'overfitting': res_gap
    }


ALL_DATASETS = ['gmsc', 'lendingclub', 'internet', 'creditcard']
all_datasets_results = {}

for ds_name in ALL_DATASETS:

    temporal_data_ds, static_data_ds, target_ds, graph_group_indices_ds, \
        group_dims_ds, T_ds, temporal_input_dim_ds = load_dataset(ds_name)

    idx_all_ds = np.arange(len(target_ds))
    idx_train_ds, idx_temp_ds, y_train_ds, y_temp_ds = train_test_split(
        idx_all_ds, target_ds, test_size=0.2, random_state=SEED, stratify=target_ds
    )
    idx_val_ds, idx_test_ds, y_val_ds, y_test_ds = train_test_split(
        idx_temp_ds, y_temp_ds, test_size=0.5, random_state=SEED, stratify=y_temp_ds
    )

    Xt_train_ds = torch.tensor(temporal_data_ds[idx_train_ds], dtype=torch.float32)
    Xt_val_ds = torch.tensor(temporal_data_ds[idx_val_ds], dtype=torch.float32)
    Xs_train_ds = torch.tensor(static_data_ds[idx_train_ds], dtype=torch.float32)
    Xs_val_ds = torch.tensor(static_data_ds[idx_val_ds], dtype=torch.float32)
    yt_train_ds = torch.tensor(y_train_ds, dtype=torch.float32)
    yt_val_ds = torch.tensor(y_val_ds, dtype=torch.float32)

    train_loader_ds = DataLoader(
        TensorDataset(Xt_train_ds, Xs_train_ds, yt_train_ds),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader_ds = DataLoader(
        TensorDataset(Xt_val_ds, Xs_val_ds, yt_val_ds),
        batch_size=BATCH_SIZE, shuffle=False
    )

    pos_rate_ds = y_train_ds.mean()
    static_dim_ds = static_data_ds.shape[1]

    ds_results = run_rq33_diagnostic(
        dataset_name=ds_name,
        train_loader=train_loader_ds,
        val_loader=val_loader_ds,
        Xt_train=Xt_train_ds,
        Xs_train=Xs_train_ds,
        yt_train=yt_train_ds,
        graph_group_indices=graph_group_indices_ds,
        temporal_input_dim=temporal_input_dim_ds,
        static_dim=static_dim_ds,
        group_dims=group_dims_ds,
        T=T_ds,
        pos_rate=pos_rate_ds,
        seed=SEED
    )

    all_datasets_results[ds_name] = ds_results

export_results_csv(all_datasets_results)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

axes[0, 0].set_visible(False)

ax = axes[0, 1]
x_pos = np.arange(len(ALL_DATASETS))
width = 0.25
loss_names = ['BCE', 'Focal_fixed (a=0.75)', 'Focal_adaptive']
bar_colors = [COLOR_PEACH_ORANGE, COLOR_LAKE_BLUE, COLOR_DEEP_CRIMSON]

for i, ln in enumerate(loss_names):
    ratios = []
    for ds_name in ALL_DATASETS:
        grad_res = all_datasets_results[ds_name]['gradient']
        if ln in grad_res:
            ratios.append(grad_res[ln]['ratio'])
        else:
            ratios.append(0)
    ax.bar(x_pos + i * width - width, ratios, width, label=ln,
           color=bar_colors[i], edgecolor='white', linewidth=0.5)

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:8] for d in ALL_DATASETS], fontsize=9)
ax.set_ylabel('Gradient Ratio', fontsize=11)
ax.set_title('(b) Gradient Balance Across Datasets', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
improvements = []
for ds_name in ALL_DATASETS:
    ls_res = all_datasets_results[ds_name]['label_smoothing']
    imp = ls_res['with_ls'][0.20]['mean'] - ls_res['no_ls'][0.20]['mean']
    improvements.append(imp)

bars = ax.bar(range(len(ALL_DATASETS)), improvements,
              color=[COLOR_DEEP_SEA if v > 0 else COLOR_BRICK_RED for v in improvements],
              edgecolor='white', linewidth=0.8, width=0.6)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(ALL_DATASETS)))
ax.set_xticklabels([d[:8] for d in ALL_DATASETS], fontsize=9)
ax.set_ylabel('AUC Improvement', fontsize=11)
ax.set_title('(c) Label Smoothing Benefit at 20% Noise', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(improvements):
    ax.text(i, v + 0.001 if v > 0 else v - 0.003, f'{v:+.4f}',
            ha='center', fontsize=9, fontweight='bold')

ax = axes[1, 1]
gaps_no = [all_datasets_results[d]['overfitting']['gap_no_ls'] for d in ALL_DATASETS]
gaps_ls = [all_datasets_results[d]['overfitting']['gap_with_ls'] for d in ALL_DATASETS]

x_pos = np.arange(len(ALL_DATASETS))
width = 0.35
ax.bar(x_pos - width / 2, gaps_no, width, label='Without LS',
       color=COLOR_BRICK_RED, edgecolor='white', linewidth=0.8)
ax.bar(x_pos + width / 2, gaps_ls, width, label='With LS',
       color=COLOR_DEEP_SEA, edgecolor='white', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:8] for d in ALL_DATASETS], fontsize=9)
ax.set_ylabel('Overfitting Gap', fontsize=11)
ax.set_title('(d) Overfitting Gap Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('RQ3.3: Loss Function Diagnostic Analysis - Cross-Dataset Comparison (3 experiments)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_SAVE_DIR, 'RQ33_cross_dataset_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()