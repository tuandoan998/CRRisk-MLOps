# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import datetime
import warnings
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from dateutil.rrule import rrule, MONTHLY
import mlflow
import joblib

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--data", type=str, help="Path to raw data")
    parser.add_argument("--train_test_ratio", type=float, help="train test splot")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
   
    args = parser.parse_args()

    return args

def get_dataframes(data_path, train_test_ratio):
    # Read CR ticket
    usecols = ['Ticket', 'Deployment Month',
           'Total Estimated Efforts', 'CTASK Number',
           'Duration', 'Tester Tested', 'Bugs', 'Incident count',
           'AVG Dev Quality Rate', 'AVG Dev Quality Rate by Exp. Year']
    df = pd.read_excel(data_path, sheet_name='CR-Data', usecols=usecols)


    df_new = pd.read_excel(data_path, sheet_name='TestingData')
    df_new = df_new[df_new['CR Stage']=='Closed']
    df_new = df_new[usecols]

    df = pd.concat([df, df_new])
    df = df.dropna()
    df['Deployment Month'] = pd.to_datetime(df['Deployment Month'])
    df = df[df['Deployment Month']>datetime.datetime(2019, 1, 1)]
    df = df.reset_index(drop=True)

    # read task data
    df_task = pd.read_excel(data_path, sheet_name='Ctask')
    df_task_new = pd.read_excel(data_path, sheet_name='TestingDataCtask',
                                usecols=set(df_task.columns)-{'Members'})
    df_task = pd.concat([df_task, df_task_new])
    df_task = df_task[df_task['CR'].isin(df['Ticket'].unique())]
    df_task = df_task.reset_index(drop=True)
    ### fill estimated effort with actual and vice versa
    df_task['Est. Effort'] = df_task.apply(lambda x: x['Est. Effort'] if x['Est. Effort']!=0 else
                                                    x['Act. Effort'], axis=1)

    df_task['Act. Effort'] = df_task.apply(lambda x: x['Act. Effort'] if x['Act. Effort']!=0 else
                                                    x['Est. Effort'], axis=1)

    ### fill 1 to the rest
    df_task['Act. Effort'] = df_task.apply(lambda x: 1 if x['Act. Effort']==0 and x['Est. Effort']==0 else
                                                        x['Act. Effort'], axis=1)

    df_task['Est. Effort'] = df_task.apply(lambda x: 1 if x['Act. Effort']==1 and x['Est. Effort']==0 else
                                                        x['Est. Effort'], axis=1)

    # read member data
    df_member = pd.read_excel(data_path, sheet_name='setup', usecols=['Member', 'Emp Code', 'Start working'])
    df["ttl_bug_inc"] = df["Bugs"] + df["Incident count"]
    df['is_bug_inc'] = df['ttl_bug_inc'].apply(lambda x: 1 if x>0 else 0)

    BASE_THRESHOLD = df['Deployment Month'].sort_values().quantile(train_test_ratio)
    base_cr_id = df[df['Deployment Month']<BASE_THRESHOLD]['Ticket'].values
    df_task_base = df_task[df_task['CR'].isin(base_cr_id)].reset_index(drop=True)

    df_task_base["ttl_bug_inc"] = df_task_base["Bugs"] + df_task_base["Incident count"]
    df_task_base['is_bug_inc'] = df_task_base['ttl_bug_inc'].apply(lambda x: 1 if x>0 else 0)

    return df, df_task, df_task_base, df_member, base_cr_id


# Ticket features
def add_ticket_features(df, df_task, df_task_base):
    related_modules_dict = df_task.groupby('CR')['Module code'].nunique().to_dict()
    df['Related modules'] = df['Ticket'].apply(lambda x: related_modules_dict[x])

    ### task level module error rate
    module_error_rate = (df_task_base.drop_duplicates(['CR', 'Module code']).groupby('Module code')['ttl_bug_inc'].sum() /
                        df_task_base.drop_duplicates(['CR', 'Module code']).groupby('Module code')['ttl_bug_inc'].count()).to_dict()
    ### CR level module error rate
    cr_error_rate_base = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        module_effort = cr_group.groupby('Module code')['Est. Effort'].sum()
        module_err_rate = np.average(a=[module_error_rate.get(item, DEFAULT_BUG_RATE) for item in module_effort.index],
                                    weights=module_effort.values)
        cr_error_rate_base[cr_id] = module_err_rate

    df_task["ttl_bug_inc"] = df_task["Bugs"] + df_task["Incident count"]
    df_task['is_bug_inc'] = df_task['ttl_bug_inc'].apply(lambda x: 1 if x>0 else 0)
    df_task = df_task.sort_values('Date Closed').reset_index(drop=True)

    ### task level module error rate
    df_cr_module_bug = df_task.sort_values('Date Closed')[['CR', 'Module code', 'ttl_bug_inc']].drop_duplicates()
    cr_module_error_rate = {}
    module_error_rate = {}
    for module in df_task['Module code'].unique():
        df_cr_module_error = df_cr_module_bug[df_cr_module_bug['Module code']==module].reset_index(drop=True)
        cumsum_error = df_cr_module_error['ttl_bug_inc'].shift().cumsum().fillna(0)
        cumsum_error_rate = cumsum_error / (cumsum_error.index+1)
        df_cr_module_error['cumsum_error_rate'] = cumsum_error_rate
        cr_module_error_rate.update(df_cr_module_error.set_index(['CR', 'Module code'])['cumsum_error_rate'].to_dict())
        module_error_rate.update(df_cr_module_error.set_index(['Module code'])['cumsum_error_rate'].to_dict())

    ### CR level module error rate
    cr_error_rate_uptodate = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        module_effort = cr_group.groupby('Module code')['Est. Effort'].sum()
        module_err_rate = np.average(a=[cr_module_error_rate.get((cr_id, module), DEFAULT_BUG_RATE) for module in module_effort.index],
                                    weights=module_effort.values)
        cr_error_rate_uptodate[cr_id] = module_err_rate

    df['Module error rate'] = df['Ticket'].apply(lambda x: cr_error_rate_base[x] if x in df_task_base['CR'].unique() else
                                                        cr_error_rate_uptodate[x])

    return df, df_task, module_error_rate, cr_error_rate_uptodate

# Dev Features
def add_dev_features(df, df_member, df_task):
    related_members_dict = df_task.groupby('CR')['Dev Emp Cd'].nunique().to_dict()
    df['Related members'] = df['Ticket'].apply(lambda x: related_members_dict[x])

    # Dev exp by months
    member_startdate_dict = df_member.set_index('Emp Code')['Start working'].dropna().to_dict()
    def get_dev_clv_exp(task):
        try:
            start_date = member_startdate_dict[task['Dev Emp Cd']]
            clv_exp = (len(list(rrule(MONTHLY, dtstart=start_date, until=task['Start Developing'])))-1)/12
        except:
            clv_exp = 0
        return clv_exp
    df_task['CLV exp'] = df_task.apply(get_dev_clv_exp, axis=1)
    ### Weighted Average exp
    cr_weighted_exp_year = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        emp_weighted_exp_year = np.average(a=cr_group.groupby('Dev Emp Cd')['CLV exp'].mean().sort_index().values,
                                        weights=cr_group.groupby('Dev Emp Cd')['Est. Effort'].sum().sort_index().values)
        cr_weighted_exp_year[cr_id] = emp_weighted_exp_year
    df['Weighted Average Exp.'] = df['Ticket'].apply(lambda x: cr_weighted_exp_year[x])

    # Dev exp by number of tickets
    cr_emp_exp_ticket = {}
    ## For each employee
    for emp_code in df_task['Dev Emp Cd'].unique():
        df_tmp_emp_exp_ticket = df_task[df_task['Dev Emp Cd']==emp_code][['CR', 'Dev Emp Cd']].drop_duplicates().reset_index(drop=True)
        ## Use cumsum to count tickets in the past of this employee
        df_tmp_emp_exp_ticket['Num Ticket'] = 1
        df_tmp_emp_exp_ticket['Num Ticket'] = df_tmp_emp_exp_ticket['Num Ticket'].shift().cumsum().fillna(0)
        cr_emp_exp_ticket.update(df_tmp_emp_exp_ticket.set_index(['CR', 'Dev Emp Cd'])['Num Ticket'].to_dict())
    ### Weighted Experience by ticket count
    cr_weighted_exp_ticket = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        dev_effort = cr_group.groupby('Dev Emp Cd')['Est. Effort'].sum()
        emp_exp_ticket = np.average(a=[cr_emp_exp_ticket.get((cr_id, emp_code), DEFAULT_BUG_RATE) for emp_code in dev_effort.index],
                                    weights=dev_effort.values)
        cr_weighted_exp_ticket[cr_id] = emp_exp_ticket
    ### Experience by ticket count
    cr_exp_ticket = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        dev_effort = cr_group.groupby('Dev Emp Cd')['Est. Effort'].sum()
        emp_exp_ticket = np.mean([cr_emp_exp_ticket.get((cr_id, emp_code), DEFAULT_BUG_RATE) for emp_code in dev_effort.index])
        cr_exp_ticket[cr_id] = emp_exp_ticket
    df['Weighted Average Ticket Count'] = df['Ticket'].apply(lambda x: cr_weighted_exp_ticket[x])
    df['Average Ticket Count'] = df['Ticket'].apply(lambda x: cr_exp_ticket[x])

    # Dev exp by number of tickets of same module
    cr_emp_module_exp_ticket = {}
    for i, row in df_task[['Dev Emp Cd', 'Module code']].drop_duplicates().iterrows():
        emp_code = row['Dev Emp Cd']
        module_code = row['Module code']
        df_tmp_emp_module_exp_ticket = df_task[(df_task['Dev Emp Cd']==emp_code)&
                                            (df_task['Module code']==module_code)][['CR', 'Dev Emp Cd', 'Module code']].drop_duplicates().reset_index(drop=True)
        df_tmp_emp_module_exp_ticket['Num Ticket'] = 1
        df_tmp_emp_module_exp_ticket['Num Ticket'] = df_tmp_emp_module_exp_ticket['Num Ticket'].shift().cumsum().fillna(0)
        cr_emp_module_exp_ticket.update(df_tmp_emp_module_exp_ticket.set_index(['CR', 'Dev Emp Cd', 'Module code'])['Num Ticket'].to_dict())
    ### Weighted Experience by ticket of same module count
    cr_weighted_module_exp_ticket = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        module_effort = cr_group.groupby(['Dev Emp Cd', 'Module code'])['Est. Effort'].sum()
        emp_module_exp_ticket = np.average(a=[cr_emp_module_exp_ticket.get((cr_id, emp_code, module_code), DEFAULT_BUG_RATE)
                                            for emp_code, module_code in module_effort.index],
                                        weights=module_effort.values)
        cr_weighted_module_exp_ticket[cr_id] = emp_module_exp_ticket
    df['Weighted Average Module Ticket Count'] = df['Ticket'].apply(lambda x: cr_weighted_module_exp_ticket[x])
    dict_module_ticket_count_error = df.groupby(pd.cut(df['Weighted Average Module Ticket Count'], range(0, 71, 5)))['ttl_bug_inc'].sum().to_dict()
    def get_module_ticket_count_error(val):
        for k, v in dict_module_ticket_count_error.items():
            if val in k:
                return v
        return 0
    df['Module Ticket Count Error rate'] = df['Weighted Average Module Ticket Count'].apply(get_module_ticket_count_error)

    return df, df_task


# Other features
def add_other_features(df, base_cr_id):
    df_base = df[df['Ticket'].isin(base_cr_id)].reset_index(drop=True)
    dict_deployment_month_error_rate = (df_base.groupby(df_base['Deployment Month'].dt.month)['is_bug_inc'].sum() /
                                        df_base.groupby(df_base['Deployment Month'].dt.month)['Ticket'].count()).to_dict()
    df['Deployment Month Error Rate'] = df['Deployment Month'].apply(lambda x: dict_deployment_month_error_rate[x.month])
    return df


# UI features
def add_ui_features(df, df_task):
    df_task['UI_Join'] = df_task[['Criterion 2', 'Criterion 3','Criterion 4']].fillna('').apply(lambda x: '/'.join(x), axis=1)

    df_cr_ui_bug = df_task.sort_values('Date Closed')[['CR', 'UI_Join', 'ttl_bug_inc']].drop_duplicates()
    cr_ui_error_rate = {}
    for ui in df_task['UI_Join'].unique():
        df_cr_ui_error = df_cr_ui_bug[df_cr_ui_bug['UI_Join']==ui].reset_index(drop=True)
        cumsum_error = df_cr_ui_error['ttl_bug_inc'].shift().cumsum().fillna(0)
        cumsum_error_rate = cumsum_error / (cumsum_error.index+1)
        df_cr_ui_error['cumsum_error_rate'] = cumsum_error_rate
        cr_ui_error_rate.update(df_cr_ui_error.set_index(['CR', 'UI_Join'])['cumsum_error_rate'].to_dict())

    ### CR level module error rate
    cr_error_rate_uptodate = {}
    for cr_id, cr_group in df_task.groupby('CR'):
        ui_effort = cr_group.groupby('UI_Join')['Est. Effort'].sum()
        ui_err_rate = np.average(a=[cr_ui_error_rate.get((cr_id, ui), DEFAULT_BUG_RATE) for ui in ui_effort.index],
                                    weights=ui_effort.values)
        cr_error_rate_uptodate[cr_id] = ui_err_rate

    df['UI error rate'] = df['Ticket'].apply(lambda x: cr_error_rate_uptodate[x])

    return df, df_task

def main(args):
    """Main function of the script."""

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    # credit_df = pd.read_csv(args.data, header=1, index_col=0)
    df, df_task, df_task_base, df_member, base_cr_id = get_dataframes(args.data, args.train_test_ratio)
    df, df_task, module_error_rate, cr_error_rate_uptodate = add_ticket_features(df, df_task, df_task_base)
    df, df_task = add_dev_features(df, df_member, df_task)
    df = add_other_features(df, base_cr_id)
    df, df_task = add_ui_features(df, df_task)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", len(FEATURE_COLS))

    x_train = df[df['Ticket'].isin(base_cr_id)][FEATURE_COLS].values
    y_train = df[df['Ticket'].isin(base_cr_id)][LABEL_COL]
    x_test = df[~df['Ticket'].isin(base_cr_id)][FEATURE_COLS].values
    y_test = df[~df['Ticket'].isin(base_cr_id)][LABEL_COL]

    sc = StandardScaler(with_mean=False)

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    joblib.dump(sc, 'std_scaler.bin', compress=True)

    # Oversampling
    cc = RandomOverSampler(random_state=RANDOM_STATE)
    over_x_train, over_y_train = cc.fit_resample(x_train, y_train)

    train_df = pd.DataFrame(over_x_train, columns = FEATURE_COLS)
    train_df['is_bug_inc'] = over_y_train.values
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df = pd.DataFrame(x_test, columns = FEATURE_COLS)
    test_df['is_bug_inc'] = y_test.values
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.data}",
        f"Train test ratio: {args.train_test_ratio}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()

    