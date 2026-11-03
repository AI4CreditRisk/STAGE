# -*- coding: utf-8 -*-

import os
import time
import copy
import math
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
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
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[2] / 'data'
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
# ---- Dataset selection: 'lendingclub' / 'gmsc' / 'internet' / 'creditcard' ----
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

FILE_MAP = {
    "lendingclub": str(DATA_DIR / "LoanStatus_2017Q1_ml_01.csv"),
    "gmsc": str(DATA_DIR / "cs_training_fe_01.csv"),
    "internet": str(DATA_DIR / "train_internetdata3.csv"),
    "creditcard": str(DATA_DIR / "default of credit card clientsUCI.csv"),
}

def load_dataset(dataset_name):
    if dataset_name not in FILE_MAP:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
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
                raise ValueError(f"Target column {target_col} not found. Current columns: {df.columns.tolist()}")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    elif dataset_name == 'internet':
        df = pd.read_csv(input_file, low_memory=False)
        target_col = 'is_default'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(np.int64)
        leakage_cols = [
            'loan_id', 'user_id', 'title', 'policy_code',
            'early_return', 'early_return_amount', 'early_return_amount_3mon'
        ]
        drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    else:  # uci
        try:
            df = pd.read_excel(input_file, skiprows=1)
        except Exception:
            df = pd.read_csv(input_file, skiprows=1)
        target_col = 'default payment next month'
        if target_col not in df.columns:
            cand = [c for c in df.columns if 'default' in c.lower()]
            if len(cand) == 0:
                raise ValueError("Default label column not found")
            target_col = cand[0]
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df[df[target_col].isin([0, 1])].copy()
        df[target_col] = df[target_col].astype(np.int64)

    # Basic preprocessing
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

    else:  # uci
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

    # Temporal augmentation: 5-window pseudo time series
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
        w1 = ['age', 'n_dependents', 'income_per_dep']
        w2 = ['open_credit_lines', 'real_estate_loans', 'credit_mix']
        w3 = ['monthly_income', 'debt_ratio', 'debt_amount_est']
        w4 = ['n30_59', 'n60_89', 'n90_late',
              'delinq_total', 'delinq_severity',
              'n30_59_flag', 'n60_89_flag', 'n90_late_flag']
        w5 = ['revolving_util']
    elif dataset_name == 'internet':
        w1 = ['work_year', 'house_exist',
              'known_dero', 'pub_dero_bankrup', 'known_outstanding_loan']
        w2 = []
        w3 = ['app_type', 'initial_list_status', 'censor_status']
        w4 = []
        w5 = ['total_loan', 'interest', 'monthly_payment', 'debt_loan_ratio',
              'recircle_b', 'recircle_util',
              'marriage', 'offsprings',
              'f0', 'f1', 'f2', 'f3', 'f4']
    else:  # uci
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

    temporal_data = np.stack(temporal_seqs, axis=1)  # (N,5,D)

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
        raise ValueError("Static feature list is empty. Please check the field mapping.")

    static_data = df[static_cols].values.astype(np.float32)

    def pick(cols):
        return [c for c in cols if c in static_cols]
    if dataset_name == 'lendingclub':
        graph_groups_def = {
            'geo': ['addr_state', 'zip_code', 'home_ownership'],
            'credit': ['grade', 'sub_grade'],
            'finance': ['annual_inc', 'dti', 'revol_util'],
            'loan': ['loan_amnt', 'term', 'purpose', 'int_rate'],
        }
    elif dataset_name == 'gmsc':
        graph_groups_def = {
            'demographic': pick(['age', 'n_dependents', 'income_per_dep']),
            'credit_structure': pick(['open_credit_lines', 'real_estate_loans', 'credit_mix']),
            'financial_pressure': pick(['monthly_income', 'debt_ratio', 'debt_amount_est']),
            'utilization': pick(['revolving_util']),
        }
    elif dataset_name == 'internet':
        def pick_static(cols):
            return [c for c in cols if c in static_cols]
        graph_groups_def = {
            'behavior': pick_static(['f0', 'f1', 'f2', 'f3', 'f4', 'f5']),
            'personal': pick_static(['house_ownership', 'house_loan_status', 'marriage', 'offsprings']),
            'geo': pick_static(['post_code', 'region']),
            'work': pick_static(['work_type', 'employment_type', 'industry']),
        }
    else:
        graph_groups_def = {
            'geo': ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE'],
            'credit': ['LIMIT_BAL'],
            'finance': ['bill_mean', 'bill_std', 'payamt_mean', 'payamt_std'],
            'loan': ['pay_max', 'pay_mean', 'pay_delay_cnt', 'pay_trend'],
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

class EpochTimer:
    def __init__(self):
        self.times = []
        self._t0 = None
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.times.append(dt)
        return dt
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0.0

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
epoch_timer = EpochTimer()
LABEL_SMOOTH = 0.05

train_loss_curve = []
val_loss_curve = []

for epoch in range(1, EPOCHS + 1):
    epoch_timer.start()
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
    if epoch <= 5:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
    model.eval()
    val_probs, val_true, val_losses = [], [], []
    with torch.no_grad():
        for xb_t, xb_s, yb in val_loader:
            xb_t, xb_s, yb = xb_t.to(DEVICE), xb_s.to(DEVICE), yb.to(DEVICE)
            prob, gate_w = model(xb_t, xb_s, graph_group_indices, return_weights=True)
            val_probs.append(prob.detach().cpu().numpy())
            val_true.append(yb.cpu().numpy())
            yb_smooth = yb * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            loss_c = criterion(prob, yb_smooth)
            loss_g = gate_entropy_regularization(gate_w)
            val_losses.append((loss_c + GATE_ENTROPY_LAMBDA * loss_g).item())
    val_probs = np.concatenate(val_probs)
    val_true = np.concatenate(val_true)
    val_auc = roc_auc_score(val_true, val_probs)
    val_loss = np.mean(val_losses)
    epoch_timer.end()
    train_loss_curve.append(np.mean(train_losses))
    val_loss_curve.append(val_loss)
    early_stopper(-val_loss, model)
    if early_stopper.early_stop:
        break

if early_stopper.best_state is not None:
    model.load_state_dict(early_stopper.best_state)

def find_best_threshold(y_true, y_prob, metric='f1'):
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= thr).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = recall_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        'AUC': auc, 'AP': ap, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'Gini': gini, 'KS': ks,
        'Type I Error': type1, 'Type II Error': type2,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'CM': cm
    }

model.eval()
val_probs, val_true = [], []
with torch.no_grad():
    for xb_t, xb_s, yb in val_loader:
        xb_t = xb_t.to(DEVICE)
        xb_s = xb_s.to(DEVICE)
        prob = model(xb_t, xb_s, graph_group_indices).detach().cpu().numpy()
        val_probs.append(prob)
        val_true.append(yb.numpy())
val_probs = np.concatenate(val_probs)
val_true = np.concatenate(val_true)
best_thr, best_f1 = find_best_threshold(val_true, val_probs, metric='f1')

test_probs, test_true = [], []
with torch.no_grad():
    for xb_t, xb_s, yb in test_loader:
        xb_t = xb_t.to(DEVICE)
        xb_s = xb_s.to(DEVICE)
        prob = model(xb_t, xb_s, graph_group_indices).detach().cpu().numpy()
        test_probs.append(prob)
        test_true.append(yb.numpy())
test_probs = np.concatenate(test_probs)
test_true = np.concatenate(test_true)
metrics_05 = compute_metrics(test_true, test_probs, threshold=0.5)
metrics_best = compute_metrics(test_true, test_probs, threshold=best_thr)
# ============================================================
# CP-Ablation diagnostic plot
# ============================================================
from scipy import stats

_DATASET_DISPLAY = {
    'lendingclub': 'LendingClub',
    'gmsc'       : 'GMSC',
    'internet'   : 'ChinesePersonalLoan',
    'creditcard' : 'UCI',
}

@torch.no_grad()
def _cp_ablation(model, Xt, Xs, y, group_indices, device, batch_size=256):
    model.eval()
    N = Xt.shape[0]
    deltas = np.zeros(N, dtype=np.float32)

    for i in range(0, N, batch_size):
        xt = Xt[i:i+batch_size].to(device)
        xs = Xs[i:i+batch_size].to(device)

        p_full = model(xt, xs, group_indices).detach()

        diff = (xt[:, 1:] - xt[:, :-1]).abs().sum(dim=-1)
        tau = diff.argmax(dim=1) + 1

        xt_masked = xt.clone()
        mean_seq = xt.mean(dim=1)
        idx = torch.arange(xt.size(0), device=device)
        xt_masked[idx, tau] = mean_seq

        p_mask = model(xt_masked, xs, group_indices).detach()

        deltas[i:i+batch_size] = (p_full - p_mask).abs().cpu().numpy()

    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    return deltas, y_np


def _plot_cp_violin(deltas, labels, display_name, save_path):
    d_pos = deltas[labels == 1]
    d_neg = deltas[labels == 0]
    med_pos = float(np.median(d_pos)) if len(d_pos) else 0.0
    med_neg = float(np.median(d_neg)) if len(d_neg) else 0.0
    selectivity = (med_pos / med_neg) if med_neg > 1e-12 else float('nan')
    try:
        _, pval = stats.mannwhitneyu(d_pos, d_neg, alternative='two-sided')
    except Exception:
        pval = float('nan')

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot([d_pos, d_neg], showmeans=False,
                          showmedians=True, widths=0.8)
    colors = ['#d9736a', '#6a91b3']
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.75); pc.set_edgecolor('black')
    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if key in parts:
            parts[key].set_color('#1f4e79'); parts[key].set_linewidth(1.2)

    ax.text(1, med_pos, f'{med_pos:.4f}', ha='center', va='bottom',
            color='#8b1a1a', fontsize=11, fontweight='bold')
    ax.text(2, med_neg, f'{med_neg:.4f}', ha='center', va='bottom',
            color='#1f4e79', fontsize=11, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Default', 'Non-Default'], fontsize=12)
    ax.set_ylabel(r'|$\Delta$p| after removing $\tau$-C', fontsize=12)
    ax.set_title(f'Change-Point Ablation\n{display_name}', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    ax.text(0.55, 0.92, f'Selectivity = {selectivity:.2f}×\n',
            transform=ax.transAxes, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#fff8dc', edgecolor='#bdb76b'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


_display = _DATASET_DISPLAY.get(str(DATASET).lower(), str(DATASET))
_deltas, _labels = _cp_ablation(model, Xt_test, Xs_test, yt_test,
                                graph_group_indices, DEVICE)
_save_path = f'./cp_ablation_{_display}.png'
_plot_cp_violin(_deltas, _labels, _display, _save_path)

# ============================================================
# 6. Full spatiotemporal diagnosis (RecentAttn + CP Selectivity + NRS)
# ============================================================
import json

@torch.no_grad()
def _recent_attn_ratio(model, Xt, Xs, group_indices, device, batch_size=256):
    """
    Extract the pooling weights from the last temporal encoder layer and
    compute the average attention ratio assigned to the last time step.
    """
    model.eval()
    ratios = []
    for i in range(0, Xt.shape[0], batch_size):
        xt = Xt[i:i+batch_size].to(device)
        h = model.temporal_encoder.embed(xt) + model.temporal_encoder.pos[:, :xt.size(1), :]
        for blk in model.temporal_encoder.layers:
            h = blk['n1'](h + blk['attn'](h))
            h = blk['n2'](h + blk['ffn'](h))
        w = torch.softmax(model.temporal_encoder.pool(h).squeeze(-1), dim=-1)  # (B, T)
        ratios.append(w[:, -1].cpu().numpy())  # weight of the last time step
    ratios = np.concatenate(ratios)
    return float(ratios.mean())


@torch.no_grad()
def _nrs_score(model, Xt, Xs, y, group_indices, device, batch_size=256, n_shuffle=3):
    """
    Neighbor Rewiring Sensitivity:
    Randomly shuffle the static feature rows (breaking graph neighbor structure)
    and measure the prediction drift ratio between positive and negative samples.
    NRS = median(|Delta p| | y=1) / median(|Delta p| | y=0)
    """
    model.eval()
    N = Xt.shape[0]
    deltas = np.zeros(N, dtype=np.float32)

    for i in range(0, N, batch_size):
        xt = Xt[i:i+batch_size].to(device)
        xs = Xs[i:i+batch_size].to(device)
        p_full = model(xt, xs, group_indices).detach()

        # Average over multiple shuffles to stabilize NRS
        p_shuffs = []
        for _ in range(n_shuffle):
            perm = torch.randperm(xs.size(0), device=device)
            xs_shuf = xs[perm]
            p_shuffs.append(model(xt, xs_shuf, group_indices).detach())
        p_mask = torch.stack(p_shuffs).mean(0)

        deltas[i:i+batch_size] = (p_full - p_mask).abs().cpu().numpy()

    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    med_pos = float(np.median(deltas[y_np == 1])) if (y_np == 1).any() else 0.0
    med_neg = float(np.median(deltas[y_np == 0])) if (y_np == 0).any() else 0.0
    nrs = med_pos / med_neg if med_neg > 1e-12 else float('nan')
    return nrs, med_pos, med_neg


def _cp_selectivity(deltas, labels):
    d_pos = deltas[labels == 1]
    d_neg = deltas[labels == 0]
    med_pos = float(np.median(d_pos)) if len(d_pos) else 0.0
    med_neg = float(np.median(d_neg)) if len(d_neg) else 0.0
    sel = med_pos / med_neg if med_neg > 1e-12 else float('nan')
    try:
        _, pval = stats.mannwhitneyu(d_pos, d_neg, alternative='two-sided')
    except Exception:
        pval = float('nan')
    return sel, pval


# ---- Compute the three main indicators ----
recent_attn = _recent_attn_ratio(model, Xt_test, Xs_test, graph_group_indices, DEVICE)
cp_sel, cp_pval = _cp_selectivity(_deltas, _labels)
nrs, nrs_pos, nrs_neg = _nrs_score(model, Xt_test, Xs_test, yt_test,
                                    graph_group_indices, DEVICE)

# ---- Determine the dominant channel ----
if cp_sel > 1.3 and nrs > 1.3:
    dominant = "Dual-channel synergy"
elif cp_sel > 1.3 or recent_attn > 0.75:
    dominant = "Temporal-dominant" if nrs < 1.3 else "Dual-channel balanced"
elif nrs > 2.0:
    dominant = "Spatial-dominant"
else:
    dominant = "Dual-channel balanced"

result = {
    'Dataset'      : _display,
    'RecentAttn'   : f"{recent_attn*100:.1f}%",
    'CP_Selectivity': f"{cp_sel:.2f}×",
    'CP_pvalue'    : f"{cp_pval:.2e}",
    'NRS'          : f"{nrs:.2f}×",
    'NRS_pos'      : f"{nrs_pos:.4f}",
    'NRS_neg'      : f"{nrs_neg:.4f}",
    'Dominant'     : dominant,
}

print("\n" + "="*70)
print(f"Spatiotemporal diagnosis results [{_display}]")
print("="*70)
for k, v in result.items():
    print(f"  {k:20s}: {v}")

# ---- Append to the summary CSV (accumulated across datasets) ----
summary_path = './spatiotemporal_diagnosis_summary.csv'
row_df = pd.DataFrame([result])
if os.path.exists(summary_path):
    old = pd.read_csv(summary_path)
    old = old[old['Dataset'] != _display]  # overwrite previous record with the same name
    all_df = pd.concat([old, row_df], ignore_index=True)
else:
    all_df = row_df
all_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(all_df.to_string(index=False))