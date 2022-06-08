import numpy             as np # linear algebra
import pandas            as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn           as sns
import matplotlib.pyplot as plt


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh")
        plt.title("Pourcentage de valeurs manquantes par feature")
        plt.show()
    else:
        print('No NAs found')



	
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



def replace_neg_val(df):
    numeric_types = ['int8', 'int32', 'int64', 'float16', 'float32', 'float64']
    for i in df.columns:
        if (df[i].dtypes in numeric_types) :
            df.loc[df[i]<0]=0
    return df

#Nous allons procéder à l'imputation des valeurs manquantes à l'aide de cette règle:
#* Pour les valeurs numérique, on utilise la moyenne
#* Pour les valeurs catégorielle, on remplace par la valeur "Unknown" 

#This function fill missing value according to the rule mentionned previously
def fill_miss_val (df):
    for i in df.columns:
        numeric_types = ['int8', 'int32', 'int64', 'float16', 'float32', 'float64']
        if (df[i].dtypes in numeric_types) :
            df[i] = df[i].fillna(df[i].mean())
            #df[i] = df[i].replace(np.nan, df[i].mean())
            df[i] = df[i].replace(np.nan, 0)
        elif (df[i].dtypes == 'object'):
            #df[i]= df[i].fillna(df[i].mode()[0]) #utiliser un encodage exemple: XXX/fillna(-99)
            df[i]= df[i].fillna("Unknown")
    number = sum([True for idx,row in df.iterrows() if any(row.isnull())])
    print("Nombre de valeur manquantes après l'imputation: ", number) 
    return df

#This function perform label encoding
def encode_df (df):
    colName = []

    for i in df.columns:
        if (df[i].dtypes == 'object'):
            colName.append(i)
    one_hot_encoded_data = pd.get_dummies(df, columns = colName)
    
    return one_hot_encoded_data



# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_classif, k=120)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs

