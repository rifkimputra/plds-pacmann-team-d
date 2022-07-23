import pandas as pd
import numpy as np
import joblib

from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split

from utils import read_yaml

LOAD_SPLIT_CONFIG_PATH = "../config/load_split_config.yaml"

def read_data(path,
              set_index = None):
    '''
    Read data from data folder in csv format.
    
    Parameters
    ----------
    path: str
          path to data
    
    '''
    
    data = pd.read_csv(path, index_col = set_index)

    return data

def split_input_output(dataset,
                       target_column):
    
    output_df = dataset[target_column]
    input_df = dataset.drop([target_column],
                            axis = 1)

    return output_df, input_df

def exclude_di(input_data):
    
    data = input_data.copy()
    data = data[data['Type of treatment - IVF or DI'] != 'DI']
    
    return data

def select_feats(input_data, selected_cols):
    
    data = input_data.copy()
    data = data[selected_cols]
    
    return data

def main_data(input_data, selected_cols):
    
    data = exclude_di(input_data)
    data = select_feats(data, selected_cols)
    
    return data

def impute_target(input_data, target_col):
    
    data = input_data.copy()
    data[target_col] = data[target_col].fillna(0)
    
    return data

def split_train_test(x, y, TEST_SIZE):
    # Do not forget to stratify if classification
    x_train, x_test,\
        y_train, y_test = train_test_split(x,
                                           y,
                                           test_size=TEST_SIZE,
                                           random_state=123,
                                           stratify=y)

    return x_train, x_test, y_train, y_test


def split_data(data_input, data_ouput, TEST_SIZE=0.2):

    x_train, x_test, \
        y_train, y_test = split_train_test(
            data_input,
            data_ouput,
            TEST_SIZE)

    x_train, x_valid, \
        y_train, y_valid = split_train_test(
            x_train,
            y_train,
            TEST_SIZE)

    return x_train, y_train, \
        x_valid, y_valid, \
        x_test, y_test

def main_load(params):
    df1 = read_data(params['file_loc1'])
    df2 = read_data(params['file_loc2'])
    data = pd.concat([df1, df2])
    data = main_data(data,params['feats'])
    data = impute_target(data,params['target_column'])

    output_df, input_df = split_input_output(data,
                                         params['target_column'])

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(input_df,
                                                                output_df,
                                                                params['test_size'])
    
    joblib.dump(X_train, params["out_path"]+"x_train.pkl")
    joblib.dump(y_train, params["out_path"]+"y_train.pkl")
    joblib.dump(X_valid, params["out_path"]+"x_valid.pkl")
    joblib.dump(y_valid, params["out_path"]+"y_valid.pkl")
    joblib.dump(X_test, params["out_path"]+"x_test.pkl")
    joblib.dump(y_test, params["out_path"]+"y_test.pkl")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    params = read_yaml(LOAD_SPLIT_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid, x_test, y_test = main_load(params)

