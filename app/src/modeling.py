import pandas as pd
import numpy as np
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, plot_roc_curve, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb

from utils import read_yaml

MODEL_CONFIG_PATH = "../config/model_config.yaml"

def load_data():
    """
    Loader for feature engineered data.
    Args:
    - params(dict): modeling params.
    Returns:
    - x_train(DataFrame): inputs of train set.
    - y_train(DataFrame): target of train set.
    - x_valid(DataFrame): inputs of valid set.
    - y_valid(DataFrame): terget of valid set.
    """

    x_train_path = "../output/x_train_ready.pkl"
    y_train_path = "../output/y_train_ready.pkl"
    x_valid_path = "../output/x_valid_ready.pkl"
    y_valid_path = "../output/y_valid_ready.pkl"
    x_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)
    x_valid = joblib.load(x_valid_path)
    y_valid = joblib.load(y_valid_path)
    return x_train, y_train, x_valid, y_valid

def select_model(X_train, y_train, X_valid, y_valid, params):
    
    logreg = LogisticRegression
    rf = RandomForestClassifier
    tree = DecisionTreeClassifier
    XGB_ = xgb.XGBClassifier
    
    train_log_dict = {'model': [logreg(), rf(), tree(), XGB_()],
                      'for_tuning': [logreg, rf, tree, XGB_], 
                      'model_name': [],
                      'model_fit': [],
                      'model_score': []}
    
    #try
    for model in train_log_dict['model']:
        base_model = model
        train_log_dict['model_name'].append(base_model.__class__.__name__)
    
    for model in train_log_dict['model']:
        base_model = model
        train_log_dict['model_fit'].append(base_model.fit(X_train,y_train))
    
    for model in train_log_dict['model_fit']:
        fitted_model = model
        train_log_dict['model_score'].append((2*(roc_auc_score(y_train, fitted_model.predict_proba(X_train)[:, 1])))-1)
        
    best_model_index = train_log_dict['model_score'].index(max(train_log_dict['model_score']))
    best_model = train_log_dict['model'][best_model_index]
    best_model_ = train_log_dict['for_tuning'][best_model_index]

    print("Gini Performance Evaluation\n")
    print(f"Logistic Regression Gini : {train_log_dict['model_score'][0]}")
    print(f"Random Forest Gini       : {train_log_dict['model_score'][1]}")
    print(f"Decision Tree Gini       : {train_log_dict['model_score'][2]}")
    print(f"XGBoost Gini : {train_log_dict['model_score'][3]}")
    print('')
    print(f"Best Model : {best_model}")
   
    #hyperparameter tuning
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=69)

    # Define search space
    space_tree = dict()

    space_tree['max_depth'] = [2, 3, 5, 10, 20] #DT
    space_tree['min_samples_leaf'] = [5, 10, 20, 50, 100] #DT
    space_tree['criterion'] = ["gini", "entropy"] #DT

    # Define search
    search_tree = RandomizedSearchCV(tree(), space_tree, n_iter=30, scoring='roc_auc', n_jobs=30, cv=cv, random_state=69)

    # Execute search
    result_tree = search_tree.fit(X_train, y_train)
    
    best_params = {'max_depth': result_tree.best_params_['max_depth'],
                  'min_samples_leaf': result_tree.best_params_['min_samples_leaf'],
                  'criterion': result_tree.best_params_['criterion']}

    print('Best Score tree: %s' % ((result_tree.best_score_ * 2) - 1))
    print('Best Hyperparameters: %s' % result_tree.best_params_)
    
    model_ = best_model_(max_depth= best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], criterion = best_params['criterion']).fit(X_train,y_train)
    
    def evaluate(true,predicted):
        f1 = f1_score(true,predicted)
        roc_auc = roc_auc_score(true,predicted)
    
        return f1,roc_auc
    
    f1, roc_auc = evaluate(y_valid, model_.predict(X_valid))
    
    print("F1 Score: ", f1)
    print("ROC AUC Score: ", roc_auc)
    
    joblib.dump(model_, params['out_path']+'best_model.pkl')
    joblib.dump(train_log_dict, params['out_path']+'train_log.pkl')
    
    return model_

if __name__ == "__main__":
    param_model = read_yaml(MODELING_CONFIG_PATH)
    x_train_ready, y_train_ready, x_valid_ready, y_valid_ready = load_fed_data()
    best_model = select_model( x_train_ready, y_train_ready, x_valid_ready, y_valid_ready, param_model)
    