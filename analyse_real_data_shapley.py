#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:29:25 2022

@author: temuuleu
"""

import warnings
from collections import Counter
import numpy as np
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE,RandomOverSampler
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV
import shap
from interpret.blackbox import ShapKernel
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


# from interpret.glassbox import (LogisticRegression,
#                                 ClassificationTree, 
#                                 ExplainableBoostingClassifier)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import f1_score, accuracy_score

from interpret import show
from interpret import preserve
from datetime import datetime

from sklearn.model_selection import train_test_split
# pip install keras

# pip install tensorflow
from library.learning_lib import create_dir,force_plot_true,shorten_names,collect_hot_encoded_features_kernel
from library.learning_lib import recreate,undummify


from library.preprocess import get_ratio, collect_hot_encoded_features,balance_age_sex

import os


def get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                     y_predicted_sample_data_test,
                     y_test_sample_data_test,
                     all_feature_names
                     ):
    
    result_df                               = pd.DataFrame( )  

    ###########
    single_index= 0
    
    for s in range(sample_size):
        
        single_explain_df = pd.DataFrame(index = (single_index,))

        if type(shapley_values) == shap._explanation.Explanation:
            feature_scores     =  shapley_values[s].values
        else:
            feature_scores     =  shapley_values[s]
        feature_values     =  shap_sample_data_test.loc[s,:]

        single_explain_df["predicted"]    = int(y_predicted_sample_data_test.iloc[s,:].values)
        single_explain_df["actual"]       =int(y_test_sample_data_test.iloc[s])
    
        for feature_name,feature_score,feature_value  in zip(all_feature_names, feature_scores,feature_values):
            
            #print(f"{ebm_feature_name:>25}  {round(feature_score,2):>20}")
            single_explain_df["score_"+feature_name] = feature_score
            single_explain_df["value_"+feature_name] = feature_value

        result_df = pd.concat([result_df, single_explain_df])
        single_index+=1
        
    return result_df



max_iteration     = 100
SEED              = 7
shapley_steps     = 5


iter_imp  = IterativeImputer(verbose=0,
                         max_iter=max_iteration,
                         tol=1e-10,
                         imputation_order='roman',
                         min_value = 0,
                         random_state=SEED)

num_imp = SimpleImputer(strategy="mean")
cat_imp = SimpleImputer(strategy="most_frequent")

data_path           = "data/real_data/selected_data.xlsx"
real_data_path      = "data/real_data/" 

real_data_df        = pd.read_excel(data_path, index_col=0).reset_index(drop=True)


wb_result_path = "result/real_data/"

create_dir(wb_result_path)

all_column          = list(real_data_df.columns)

label_columns       = 'mRS_a_neu'



columns_1   = ['hsCRP', 'pmchol', 'pmglu', 'scbig_a', 'age', 'i_bi_y']
columns_2   = ['hsCRP', 'pmchol', 'pmglu', 'age', 'i_bi_y']
columns_3   = ['pmchol', 'pmglu', 'age', 'i_bi_y']

columns = [columns_1,columns_2,columns_3]

categorical_columns   = ['sex',
                   'hxsmok_a',
                   's_hxalk_a',
                   's_kltoas_TL2',
                    'hxchol_ps']


multip_categorical_columns =  ['hxsmok_a','s_kltoas_TL2']


all_result_df = pd.DataFrame()
all_emb_result_df = pd.DataFrame()

gender_balance = [1,0]
label_balance  = [1,0]

shapley_sample_sizes = [5,15,30,40,50,60]


big_small_columns       = [0,1]
seeds                   = 30
stability_index_counter = 0
model_counts            = 3

k_neighbors=5

smote = SMOTE( random_state=SEED, n_jobs=-1,k_neighbors=k_neighbors)


analyse_name = "real_data_seeds_result"


real_data_smpale_df = real_data_df
data_sample_size = real_data_df.shape[0]

step = 0
how_many_step = seeds * len(gender_balance) * len(label_balance) * shapley_steps * len(big_small_columns) * model_counts

for seed in range(seeds):
    for shap_sample_size in shapley_sample_sizes:
        for numerical_columns in columns:
            for balance in gender_balance:
                for l_balance  in label_balance:

                    print(f"balance {balance}")
                    print(f"seed {seed}")
                    
                    Y_data     =    real_data_smpale_df[label_columns]
                    X_data     =    real_data_smpale_df[numerical_columns+categorical_columns]
                    
                    columns_len = len(X_data.columns)
                    
                    ratio = Counter(Y_data)
                    porportion = get_ratio(ratio[1], ratio[0])
                    
        
                    if balance == 1:
                        X_data,Y_data =  balance_age_sex(X_data,Y_data, name=str(seed)+"_black_"+str(data_sample_size), show=False)
                        
                    number_of_women = sum(X_data["sex"] == 1)
                    number_of_men   = sum(X_data["sex"] == 0)
                        
                    data_size  = X_data.shape[0]   
                    
                    print(X_data.shape)
            
                    cv_outer   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            
                    ci = 0
                    
            
                    for train_index, test_index in cv_outer.split(X_data, Y_data):
                        
                        all_full_models_df   = pd.DataFrame()                    
                        all_ci_models_df   = pd.DataFrame()

                        
                        os.system("clear")
                        print(f"seed: {seed}")
      
                        percent_complete = round((step / how_many_step)  * 100, 2)
                        print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
            
                        ci+=1
            
                        X_train, X_test = X_data.iloc[train_index, :], X_data.iloc[test_index, :]
                        y_train, y_test = pd.DataFrame(Y_data).iloc[train_index,0], pd.DataFrame(Y_data).iloc[test_index,0]   
                        
                        ci_number_of_women = sum(X_train["sex"] == 1)
                        ci_number_of_men   = sum(X_train["sex"] == 0)
                        
                        #preprocess
                        X_train_num = X_train[numerical_columns]
                        X_train_cat = X_train[categorical_columns]
            
                        X_test_num = X_test[numerical_columns]
                        X_test_cat = X_test[categorical_columns]
            
                        #missing data
                        #prepare train data
                        cat_imp.fit(X_train_cat) 
                        X_train_cat  = pd.DataFrame(cat_imp.transform(X_train_cat),
                                            columns = list(X_train_cat.columns), index=X_train_cat.index)
            
            
                        X_train_full = pd.concat([X_train_cat,X_train_num],axis = 1)
                        iter_imp.fit(X_train_full)
                        num_imp.fit(X_train_full)
            
                        X_train_full     = pd.DataFrame( np.round(iter_imp.transform(X_train_full),2), columns = X_train_full.columns)
                        #X_train_full_hot = pd.get_dummies(X_train_full, columns = categorical_columns,prefix_sep="+")
            
                        #prepare test data
            
                        X_test_cat      = pd.DataFrame(cat_imp.transform(X_test_cat),
                                          columns = list(X_test_cat.columns), index=X_test_cat.index)
            
            
                        X_test_full     = pd.concat([X_test_cat,X_test_num],axis = 1)
                        X_test_full     = pd.DataFrame( np.round(num_imp.transform(X_test_full),2), columns = X_train_full.columns)
                        #X_test_full_hot = pd.get_dummies(X_test_full, columns = categorical_columns,prefix_sep="+")
                        
                        
                        #onehot encoding full range
                        X_train_full["train"] = 1
                        X_test_full["train"]= 0
        
                        full_data = pd.concat([X_train_full,X_test_full ])
                        
                        full_data_hot = pd.get_dummies(full_data, columns = multip_categorical_columns,prefix_sep="+")
                        
                        X_train_full_hot = full_data_hot.loc[full_data_hot["train"] == 1,:].drop(columns=["train"])
                        X_test_full_hot = full_data_hot.loc[full_data_hot["train"] == 0,:].drop(columns=["train"])
                        y_test = y_test.reset_index(drop=True )
                        
                        
                        all_feature_names = list(X_train_full_hot.columns)
                        
                        #if not X_train_full_hot.shape[1] == X_test_full_hot.shape[1]: continue
                    
        
                        #tree model #######################################################
                        
                        if l_balance == 1:    
                            X_train_full_hot, y_train = smote.fit_resample(X_train_full_hot,y_train)
                            
                        ratio = Counter(y_train)
                        porportion = get_ratio(ratio[1], ratio[0])
                        
                        number_of_0_label =  Counter(y_train)[0]
                        number_of_1_label =  Counter(y_train)[1]
        

                        scaler = StandardScaler()
                        scaler.fit(X_train_full_hot)
                        X_train_std = scaler.transform(X_train_full_hot)
                        X_test_std = scaler.transform(X_test_full_hot)
            
                        linear_model  = LogisticRegression(penalty="l2", C=0.1)
                        linear_model = linear_model.fit(X_train_std, y_train)
                        model_name  = "LogisticRegression"
                        
                        y_pred= linear_model.predict(X_test_std)
                        
     
                        stability_index_counter+=1
                        
                        pred_proba_test_auc = linear_model.predict_proba(X_test_std)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)
            
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
            
                        #sensivity
                        TPR = tp / (tp + fn)
                        #specificity
                        TNR = tn / (tn + fp)
                        #balanced accuracy
                        BA = round((TPR + TNR) / 2, 2)
                        
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        print("")
                        print(f"Linear BA {BA}")
                        print(f"Linear roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print("")
                       
                        predict_fn = lambda x: linear_model.predict_proba(x)
                        # %% Create kernel shap explainer
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                 feature_names=all_feature_names)
                        
                        
                        linear_explainer   = shap.Explainer(linear_model, X_train_std, feature_names=all_feature_names, n_jobs=-1)

                        ci_model_result_df = pd.DataFrame()
                        
                        for sample_seed in range(shapley_steps):
                            
                            result_df = pd.DataFrame()
                            
                            print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
                            
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)  
                            
                            sample_size = len(shap_sample_data_test)
                            
                            shap_sample_data_test_std  = scaler.transform(shap_sample_data_test)
        
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                all_linear_shap_values = linear_explainer(shap_sample_data_test_std)
                                
                                
                                linear_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                                                                        all_linear_shap_values.values)
                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                         feature_names = all_feature_names)
                                
                            
                                kernel_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                                                                        all_kernel_shap_values[1])


                            shapley_values                          = all_linear_shap_values
                            
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)

                            temp_df["explainer"]                  = "linear_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            
                            
                            
                            
                            shapley_values                          = all_kernel_shap_values[1]
                            
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                            
                            
                            temp_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            result_df["sample_step"]                = sample_seed
                            step +=1
                            
                            
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                            
                            
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
   
  
                        # %% Tree based model
                        rf = RandomForestClassifier()
                        rf.fit(X_train_full_hot, y_train)
                        y_pred = rf.predict(X_test_full_hot)
    
                        model_name  = "RandomForestClassifier"
                        
                        
                        pred_proba_test_auc = rf.predict_proba(X_test_full_hot)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)
            
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
            
                        #sensivity
                        TPR = tp / (tp + fn)
                        #specificity
                        TNR = tn / (tn + fp)
                        #balanced accuracy
                        BA = round((TPR + TNR) / 2, 2)
                        
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        print("")
                        print(f"RandomForest {BA}")
                        print(f"roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print(f"Accuracy {round(accuracy_score(y_test, y_pred),2)}")
                        print("")
                        
        
                        # %% Create kernel shap explainer
                        predict_fn = lambda x: rf.predict_proba(x)
                        
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                  feature_names=all_feature_names)
                        
                        
                        # %% Create SHAP explainer
                        TreeExplainer = shap.TreeExplainer(rf, feature_names=all_feature_names,n_jobs=-1)

                        
                        ci_model_result_df = pd.DataFrame()
                        for sample_seed in range(shapley_steps):
                            print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
                            #sample from test
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)  
                            
                            sample_size = len(shap_sample_data_test)
                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                          feature_names = all_feature_names)
                                
                            
                                kernel_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                                                                        all_kernel_shap_values[1])
                                
                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                
                                all_tree_shap_values                    = TreeExplainer.shap_values(shap_sample_data_test)
                            
                
                                tree_shap_values, feature_names     = collect_hot_encoded_features(shap_sample_data_test,
                                                                        all_tree_shap_values[1])
                                

                            shapley_values                          = all_tree_shap_values[1]
                            
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
   
                            temp_df["explainer"]                  = "TreeExplainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            
                            
                            shapley_values                          = all_kernel_shap_values[1]
                            
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                            
                            
                            temp_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            result_df["sample_step"]                = sample_seed
                            step +=1
                            
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                         
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
                            
                            
                        # %% Fit Explainable Boosting Machine
                        ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
                        ebm.fit(X_train_full_hot, y_train)
                        
                        model_name  = "ExplainableBoostingClassifier"
                
                        y_pred = ebm.predict(X_test_full_hot)
    
                        pred_proba_test_auc = ebm.predict_proba(X_test_full_hot)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)
            
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
            
                        #sensivity
                        TPR = tp / (tp + fn)
                        #specificity
                        TNR = tn / (tn + fp)
                        #balanced accuracy
                        BA = round((TPR + TNR) / 2, 2)
                        
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        
                        print("")
                        print(f"ExplainableBoostingClassifier BA {BA}")
                        print(f"roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print("")
                        
                        # %% Create kernel shap explainer
                        predict_fn = lambda x: ebm.predict_proba(x)
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                  feature_names=all_feature_names)
                        ci_model_result_df   = pd.DataFrame()

                        for sample_seed in range(shapley_steps):
                            print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
                            
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)  
                            
                            
                            #ExplainableBoosting_internal

                            ebm_local = ebm.explain_local(shap_sample_data_test,y_test_sample_data_test)
                            ebm_feature_names = ebm_local.feature_names
        
                            size = shap_sample_data_test.shape[0]
        
                            result_df = pd.DataFrame()
                            emb_index= 0
                            
                            for s in range(size):
                            
                                single_explain_df = pd.DataFrame(index = (emb_index,))
                                
                                feature_scores = ebm_local.data(s)['scores']
                                feature_values = ebm_local.data(s)['values'] 
                                
                                actual = ebm_local.data(s)['perf']['actual']
                                predicted = ebm_local.data(s)['perf']['predicted']
                                
                                single_explain_df["actual"]    = actual
                                single_explain_df["predicted"] = predicted
                                
                                for ebm_feature_name,feature_score,feature_value  in zip(ebm_feature_names, feature_scores,feature_values):
                                    #print(f"{ebm_feature_name:>25}  {round(feature_score,2):>20}")
                                    single_explain_df["score_"+ebm_feature_name] = feature_score
                                    single_explain_df["value_"+ebm_feature_name] = feature_value
                                
                                result_df = pd.concat([result_df, single_explain_df])
                                emb_index+=1
                                
                                
                            result_df["explainer"]                  = "ExplainableBoosting_internal"  
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                            
                            
                            ##ExplainableBoosting_internal
                            
                        
                            shapley_values                          = all_kernel_shap_values[1]

                            
                            result_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                           
                            result_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df["sample_step"]                = sample_seed
                            step +=1

                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                            
 
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        ci_model_result_df["ci_number_of_men"]           = ci_number_of_men
                        ci_model_result_df["ci_number_of_women"]         = ci_number_of_women    

                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
                        

                        #collecting all result from all modell


                        all_full_models_df  = pd.concat([all_full_models_df,all_ci_models_df])            
    
                        all_full_models_df["age_sex_balance"]            = balance
                        all_full_models_df["label_balance"]              = l_balance
    
                        all_full_models_df["number_of_men"]              = number_of_men
                        all_full_models_df["number_of_women"]            = number_of_women
                        all_full_models_df["shap_sample_size"]           = shap_sample_size
                        
                        all_full_models_df["data_sample_size"]           = data_sample_size
                        all_full_models_df["data_size_after_balance"]    = data_size
                        
                        all_full_models_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                        all_full_models_df["stability_index_counter"]    = stability_index_counter
                        all_full_models_df["it_seed"]                       = seed
                        all_full_models_df["bsc"]                       = str(numerical_columns)
                        all_full_models_df["columns_len"]               = columns_len
                        
                        all_emb_result_df = pd.concat([all_emb_result_df,all_full_models_df])
                        
                        print(f"result_path {wb_result_path}")
                        
                        all_emb_result_df.to_excel(wb_result_path+f"part_{analyse_name}_result.xlsx")
                    
                    
                    all_emb_result_df.to_excel(wb_result_path+f"all_{analyse_name}_result.xlsx")
                        
                    try:   
                        all_emb_result_df.to_excel(wb_result_path+f"all_{analyse_name}_cp.xlsx")
                    except:
                        pass
                        
                                

                            
                            

                            
                            
                            
                            

                        
                        
  
                            
        
                    #         result_df["explainer"]                  = "TreeExplainer_shap"  
                    #         result_df["model_name"]                 = "RandomForestClassifier"
                            
                    #         result_df["age_sex_balance"]            = balance
                    #         result_df["label_balance"]              = l_balance
                
                    #         result_df["it_seed"]                    = seed
                    #         result_df["ci"]                         = ci
                    #         result_df["ba"]                         = round(BA,2) 
                    #         result_df["f1"]                         = round(f1,2) 
                            
                    #         result_df["sensivity"]                  = round(TPR,2)
                    #         result_df["specificity"]                = round(TNR,2)
                
                    #         result_df["tp"]                         = round(TNR,2)
                    #         result_df["tn"]                         = round(tn,2)
                    #         result_df["fp"]                         = round(fp,2)
                    #         result_df["fn"]                         = round(fn,2)
                    #         result_df["specificity"]                = round(TNR,2)
                            
                    #         result_df["roc_auc"]                    = round(roc_auc,2) 
                            
                    #         result_df["number_of_men"]              = number_of_men
                    #         result_df["number_of_women"]            = number_of_women
                    #         result_df["shap_sample_size"]           = shap_sample_size
                            
                    #         result_df["data_sample_size"]           = data_sample_size
                    #         result_df["data_size_after_balance"]    = data_size
                            
                    #         result_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                            
                    #         result_df["stability_index_counter"]    = stability_index_counter
                    #         result_df["sample_seed"]                = sample_seed
                    #         result_df["step"]                       = step
                    #         result_df["bsc"]                        = bsc
                            
                    #         result_df["blackbox"]                   = 1
    
                    #         all_emb_result_df = pd.concat([all_emb_result_df,result_df])
                            
                            
                    #         #kernelshap
                    #         shapley_values                          = all_kernel_shap_values[1]
                            
                    #         result_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                    #                              y_predicted_sample_data_test,
                    #                              y_test_sample_data_test,
                    #                              all_feature_names)
                            
                            
                    #         result_df["explainer"]                  = "kernel_explainer_shap"  
                    #         result_df["model_name"]                 = "RandomForestClassifier"
                            
                    #         result_df["age_sex_balance"]            = balance
                    #         result_df["label_balance"]              = l_balance
                
                    #         result_df["it_seed"]                    = seed
                    #         result_df["ci"]                         = ci
                    #         result_df["ba"]                         = round(BA,2) 
                    #         result_df["f1"]                         = round(f1,2) 
                            
                    #         result_df["sensivity"]                  = round(TPR,2)
                    #         result_df["specificity"]                = round(TNR,2)
                
                    #         result_df["tp"]                         = round(TNR,2)
                    #         result_df["tn"]                         = round(tn,2)
                    #         result_df["fp"]                         = round(fp,2)
                    #         result_df["fn"]                         = round(fn,2)
                    #         result_df["specificity"]                = round(TNR,2)
                            
                    #         result_df["roc_auc"]                    = round(roc_auc,2) 
                            
                    #         result_df["number_of_men"]              = number_of_men
                    #         result_df["number_of_women"]            = number_of_women
                    #         result_df["shap_sample_size"]           = shap_sample_size
                            
                    #         result_df["data_sample_size"]           = data_sample_size
                    #         result_df["data_size_after_balance"]    = data_size
                            
                    #         result_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                            
                    #         result_df["stability_index_counter"]    = stability_index_counter
                    #         result_df["sample_seed"]                = sample_seed
                    #         result_df["step"]                       = step
                    #         result_df["bsc"]                        = bsc
                            
                    #         result_df["blackbox"]                   = 1
    
                    #         all_emb_result_df = pd.concat([all_emb_result_df,result_df])
                            
                            

        
            


                   



                    #     result_df["model_name"]                 = "ExplainableBoostingClassifier"
                        
                    #     result_df["age_sex_balance"]            = balance
                    #     result_df["label_balance"]            = l_balance
            
                    #     result_df["it_seed"]                    = seed
                    #     result_df["ci"]                         = ci
                    #     result_df["ba"]                         = round(BA,2) 
                    #     result_df["f1"]                         = round(f1,2) 
                        
                    #     result_df["sensivity"]                  = round(TPR,2)
                    #     result_df["specificity"]                = round(TNR,2)
            
                    #     result_df["tp"]                         = round(TNR,2)
                    #     result_df["tn"]                         = round(tn,2)
                    #     result_df["fp"]                         = round(fp,2)
                    #     result_df["fn"]                         = round(fn,2)
                    #     result_df["specificity"]                = round(TNR,2)
                        
                    #     result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                    #     result_df["number_of_men"]              = number_of_men
                    #     result_df["number_of_women"]            = number_of_women
                        
                    #     result_df["data_sample_size"]           = data_sample_size
                    #     result_df["data_size_after_balance"]    = data_size
                        
                    #     result_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                        
                    #     result_df["stability_index_counter"]    = stability_index_counter
                    #     result_df["sample_seed"]                = sample_seed
                    #     result_df["step"]                       = step
                    #     result_df["bsc"]                       = bsc
                        
                    #     result_df["blackbox"]                   = 0

                    #     all_emb_result_df = pd.concat([all_emb_result_df,result_df])
                    #     all_emb_result_df.to_excel(wb_result_path+"all_real_data_seeds_result.xlsx")

                        
                    #     with warnings.catch_warnings():
                    #         warnings.filterwarnings("ignore")
                    #         all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                    #                                                  feature_names = all_feature_names)
                            
                        
                    #         kernel_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                    #                                                 all_kernel_shap_values[1])
                            

                    #     result_df                               = pd.DataFrame( index=(stability_index_counter,))  
                        
                        
                    #     mean_kernel_shap_values = list(np.round(np.mean(abs(kernel_shap_values), axis = 0),4))
                    #     std_kernel_shap_values = list(np.round(np.std(abs(kernel_shap_values), axis = 0),4))
        
        
                    #     for fi,feature_name in enumerate(feature_names):
                    #         result_df["kernelshap_mean_"+feature_name] = mean_kernel_shap_values[fi]
            
                    #     for fi,feature_name in enumerate(feature_names):
                    #         result_df["kernelshap_std_"+feature_name] = std_kernel_shap_values[fi]
            
        
                    #     result_df["model_name"]                 = "ExplainableBoostingClassifier"
                        
                    #     result_df["age_sex_balance"]            = balance
                    #     result_df["label_balance"]            = l_balance
            
                    #     result_df["it_seed"]                    = seed
                    #     result_df["ci"]                         = ci
                    #     result_df["ba"]                         = round(BA,2) 
                    #     result_df["f1"]                         = round(f1,2) 
                        
                    #     result_df["sensivity"]                  = round(TPR,2)
                    #     result_df["specificity"]                = round(TNR,2)
            
                    #     result_df["tp"]                         = round(TNR,2)
                    #     result_df["tn"]                         = round(tn,2)
                    #     result_df["fp"]                         = round(fp,2)
                    #     result_df["fn"]                         = round(fn,2)
                    #     result_df["specificity"]                = round(TNR,2)
                        
                    #     result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                    #     result_df["number_of_men"]              = number_of_men
                    #     result_df["number_of_women"]            = number_of_women
                        
                    #     result_df["data_sample_size"]           = data_sample_size
                    #     result_df["data_size_after_balance"]    = data_size
                        
                    #     result_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                        
                    #     result_df["stability_index_counter"]    = stability_index_counter
                    #     result_df["sample_seed"]                = sample_seed
                    #     result_df["step"]                       = step
                    #     result_df["bsc"]                       = bsc
                        
                    #     result_df["blackbox"]                   = 1
                    #     step +=1

                    #     all_result_df = pd.concat([all_result_df,result_df])
                    #     all_result_df.to_excel(bb_result_path+"all_real_data_seeds_result.xlsx")