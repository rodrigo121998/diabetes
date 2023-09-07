import dill
import os
import pandas as pd
import numpy as np

#import dill
def notebook_export_variables():
    dill.dump_session('notebook_env.db')

def notebook_import_variables():
    dill.load_session('notebook_env.db')

def import_data(main_folder,year):
    directory=os.path.join(main_folder,year)

    df = pd.DataFrame()
    counter=0

    list_months=["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

    for file in os.listdir(directory):
        if file.endswith(".xls"):
            print(counter+1, 'importando', file[27:])
            counter+=1
            temp=pd.read_excel(os.path.join(directory,file), parse_dates = True)
            temp.columns=temp.columns.str.lower()

            try:
                temp.rename(columns={'direcorate':'directorate','directorates':'directorate'},inplace=True)
            except:
                pass

            # Define a dictionary to map old column names to new trimmed names
            column_mapping = {col: col.strip().replace(' ', '_').replace('!', '_').replace('.', '_') for col in temp.columns}

            # Rename the columns
            temp = temp.rename(columns=column_mapping)

            df=df._append(temp)
            for mo in list_months:
                #if mo in filename or mo[1:3] in filename:
                if mo in file:
                    list_months.remove(mo)
    
    # df.columns=df.columns.str.lower()    

    df.dropna(how='all',inplace=True)

    print("missing months", list_months)

    return df

def mode_string(x):
    if x.mode().empty:
        return None
    return x.mode().iloc[0]

def ks_table(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)

class FrequencyEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.counts_dict = None

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        counts_dict = {}
        for col in self.cols:
            try:
                values, counts = np.unique(X[col], return_counts=True)
            except:
                values, counts = np.unique(X[col].astype('str'), return_counts=True)
            counts_dict[col] = dict(zip(values, counts))
        self.counts_dict = counts_dict

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        counts_dict_test = {}
        res = []
        for col in self.cols:
            try:
                values, counts = np.unique(X[col], return_counts=True)
            except:
                values, counts = np.unique(X[col].astype('str'), return_counts=True)
            counts_dict_test[col] = dict(zip(values, counts))

            # if value is in "train" keys - replace "test" counts with "train" counts
            for k in [key for key in counts_dict_test[col].keys() if key in self.counts_dict[col].keys()]:
                counts_dict_test[col][k] = self.counts_dict[col][k]

            res.append(X[col].map(counts_dict_test[col]).values.reshape(-1, 1))
        res = np.hstack(res)

        X[self.cols] = res
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        X = self.transform(X)
        return X

# Define a function to categorize the primary diagnosis
def categorize_diagnosis(diag):
    if any(char in diag for char in ['V', 'E']):
        return "Other"
    elif diag.startswith("250"):
        return "Diabetes"
    elif (390 <= float(diag) <= 459) or diag == "785":
        return "Circulatory"
    elif (460 <= float(diag) <= 519) or diag == "786":
        return "Respiratory"
    elif (520 <= float(diag) <= 579) or diag == "787":
        return "Digestive"
    elif (580 <= float(diag) <= 629) or diag == "788":
        return "Genitourinary"
    elif (140 <= float(diag) <= 239):
        return "Neoplasms"
    elif (710 <= float(diag) <= 739):
        return "Musculoskeletal"
    elif (800 <= float(diag) <= 999):
        return "Injury"
    else:
        return "Other"
    
def remover_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df