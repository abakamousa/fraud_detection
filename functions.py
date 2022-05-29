import numpy             as np # linear algebra
import pandas            as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn           as sns
import matplotlib.pyplot as plt


def info_missing_value (df):
    print("Number of rows of the dataset identity: ", df.shape[0])
    print("Number of columns of the dataset identity: ", df.shape[1])
    nb = sum([True for idx,row in df.iterrows() if any(row.isnull())])
    print("Number of missing values in the dataset : ", nb)
    percentage_missing_id = (nb/(df.shape[0]*df.shape[1]))*100
    print(percentage_missing_id, "% values of the dataset is missing" )
	
	
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

def outlier_detection(df):
    for i in df.columns:
        upper = df[i].mean() + 3*df[i].std()
        lower = df[i].mean() - 3*df[i].std()
        df = df[(df[i]<upper) & (df[i]>lower)]
        return df
		
def plot_missing_value_features (df):
    count1= df.isnull().sum()
    plt.rcParams["figure.figsize"] = (23,8)
    fig = sns.barplot(x=count1.index, y=count1)
    fig.set(xlabel="Feature", ylabel='Nombre de valeurs manquantes')
    plt.xticks(rotation = 90)
    plt.show()
	
def missing_value_per_observation (df):
    count_miss_row =[]
    for idx,row in df.iterrows():
        count = df.iloc[idx:idx+1,:].isnull().sum().sum()
        count_miss_row.append(count/df.shape[0])
    y_obs = range(len(count_miss_row))
    data = {'Observation':y_obs, 'Number_of_Missing_Val':count_miss_row}
    df_miss = pd.DataFrame(data)
    return df_miss.sort_values(by='Number_of_Missing_Val', ascending=False)

#Nous allons procéder à l'imputation des valeurs manquantes à l'aide de cette règle:
#* Pour les valeurs numérique, on utilise la moyenne
#* Pour les valeurs catégorielle, on remplace par la valeur "Unknown" 

#This function fill missing value according to the rule mentionned previously
def fill_miss_val (df):
    for i in df.columns:
        if (df[i].dtypes == 'int64')|(df[i].dtypes == 'float64') :
            df[i]= df[i].fillna(df[i].mean())
        elif (df[i].dtypes == 'object'):
            #df[i]= df[i].fillna(df[i].mode()[0]) #utiliser un encodage exemple: XXX/fillna(-99)
            df[i]= df[i].fillna("Unknown")
    number = sum([True for idx,row in df.iterrows() if any(row.isnull())])
    print("Nombre de valeur manquantes après le remplacement: ", number) 
    return df

#This function perform label encoding
def encode_df (df):
    colName = []

    for i in df.columns:
        if (df[i].dtypes == 'object'):
            colName.append(i)
    one_hot_encoded_data = pd.get_dummies(df, columns = colName)
    
    return one_hot_encoded_data
	
def merge_df(df1, df2, param):
    df_merged = pd.merge(df1, df2, on = param, how = "outer")
    return df_merged
	
