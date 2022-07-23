import pandas as pd
import numpy as np
import joblib

import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import time

from utils import read_yaml

ENGINEERING_CONFIG_PATH = "../config/engineering_config.yaml"

def load_split_data(params):
    """
    Loader for splitted data.
    
    Args:
    - params(dict): preprocessing params.
    
    Returns:
    - x_train(DataFrame): inputs of train set.
    - x_valid(DataFrame): inputs of valid set.
    - x_test(DataFrame): inputs of test set.
    """

    x_train = joblib.load(params["out_path"]+"x_train.pkl")
    y_train = joblib.load(params["out_path"]+"y_train.pkl")
    x_valid = joblib.load(params["out_path"]+"x_valid.pkl")
    y_valid = joblib.load(params["out_path"]+"y_valid.pkl")
    x_test = joblib.load(params["out_path"]+"x_test.pkl")
    y_test = joblib.load(params["out_path"]+"y_test.pkl")

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def to_numeric(input_data, do=True):
    
    data = input_data.copy()
    
    # replace '> 50' with 51
    data['Fresh Eggs Collected'] = data['Fresh Eggs Collected'].replace(['> 50'],[51])
    data['Eggs Mixed With Partner Sperm'] = data['Eggs Mixed With Partner Sperm'].replace(['> 50'],[51])
    
    # replace '>=5' with 6
    data['Total Number of Previous IVF cycles'] = data['Total Number of Previous IVF cycles'].replace(['>=5'],[6])
    data['Total number of IVF pregnancies'] = data['Total number of IVF pregnancies'].replace(['>=5'],[6])
    
    # convert to numerical data
    data['Fresh Eggs Collected'] = pd.to_numeric(data['Fresh Eggs Collected'])
    data['Eggs Mixed With Partner Sperm'] = pd.to_numeric(data['Eggs Mixed With Partner Sperm'])
    data['Total Number of Previous IVF cycles'] = pd.to_numeric(data['Total Number of Previous IVF cycles'])
    data['Total number of IVF pregnancies'] = pd.to_numeric(data['Total number of IVF pregnancies'])
    
    return data

def replace_age(input_data, cats, do=True):
    
    data = input_data.copy()
    data.drop(data[data['Patient Age at Treatment'] == '999'].index, inplace = True)
    data['Patient Age at Treatment'] = data['Patient Age at Treatment'].replace(cats)
    
    return data

def get_dummies(input_data, col, do=True):
    
    data = input_data.copy()
    data = pd.get_dummies(data, columns=col, prefix=col)
    
    return data

def replace_eggsrc(input_data, do=True):
    
    data = input_data.copy()
    data['Egg Source'] = data['Egg Source'].replace(['Patient','Donor'],[0,1])
    
    return data

def remove_cols(input_data, cols, do=True):
    
    data = input_data.copy()
    data = data.drop(columns=cols)
    
    return data

def undersampling(x_train, y_train):
    
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_train, y_train = undersample.fit_resample(x_train, y_train)
    
    return X_train, y_train

def preprocess(input_data,params):
    """
    A function to execute the preprocessing steps.
    
    Args:
    - df_in(DataFrame): Input dataframe
    - params(dict): preprocessing parameters
    
    Return:
    - df(DataFrame): preprocessed data
    """
    data = input_data.copy()
    data = to_numeric(data, params['to_numeric'])
    data = replace_age(data, params['age_replace'], params['replace_age'])
    data = get_dummies(data, params['to_dummy'], params['get_dummies'])
    data = replace_eggsrc(data, params['replace_eggsrc'])
    data = remove_cols(data, params['to_remove'], params['remove_cols'])
    
    return data

def main_eng(x_train, y_train, x_valid, y_valid, x_test, y_test, params):
    
    # concat data
    df_train = pd.concat([x_train, pd.DataFrame(y_train)], axis = 1)
    df_valid = pd.concat([x_valid, pd.DataFrame(y_valid)], axis = 1)
    df_test = pd.concat([x_test, pd.DataFrame(y_test)], axis = 1)
    
    df_list = [df_train, df_valid, df_test]
    df_preprocessed = []
    
    for x in df_list:
        temp = preprocess(x, params)
        df_preprocessed.append(temp)
        
    X_train_ready = df_preprocessed[0].drop(columns=['Live Birth Occurrence'], axis=1)
    y_train_ready = df_preprocessed[0]['Live Birth Occurrence']
    X_valid_ready = df_preprocessed[1].drop(columns=['Live Birth Occurrence'], axis=1)
    y_valid_ready = df_preprocessed[1]['Live Birth Occurrence']
    X_test_ready = df_preprocessed[2].drop(columns=['Live Birth Occurrence'], axis=1)
    y_test_ready = df_preprocessed[2]['Live Birth Occurrence']
    
    X_train_ready, y_train_ready = undersampling(X_train_ready, y_train_ready)
    
    joblib.dump(X_train_ready, params["out_path"]+"X_train_ready.pkl")
    joblib.dump(y_train_ready, params["out_path"]+"y_train_ready.pkl")
    joblib.dump(X_valid_ready, params["out_path"]+"X_valid_ready.pkl")
    joblib.dump(y_valid_ready, params["out_path"]+"y_valid_ready.pkl")
    joblib.dump(X_test_ready, params["out_path"]+"X_test_ready.pkl")
    joblib.dump(y_test_ready, params["out_path"]+"y_test_ready.pkl")
    
    return X_train_ready, y_train_ready, X_valid_ready, y_valid_ready, X_test_ready, y_test_ready 

if __name__ == "__main__":
    params_engineering = read_yaml(ENGINEERING_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_split_data(params_engineering)
    X_train_ready, y_train_ready, X_valid_ready, y_valid_ready, X_test_ready, y_test_ready = main_eng(x_train, 
                                                                                                      y_train, 
                                                                                                      x_valid, 
                                                                                                      y_valid, 
                                                                                                      x_test, 
                                                                                                      y_test,
                                                                                                      params_engineering)