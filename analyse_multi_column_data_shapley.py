#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:20:53 2022

@author: temuuleu
"""

import os 
from library.synthetic_creator import SingleDataCreater
import pandas as pd
import configparser
from library.learning_lib import create_dir
import argparse
from collections import Counter
import warnings

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
from imblearn.datasets import make_imbalance

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


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}
                        

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


def create_single_column_data(added_columns):
    """
    create synthetic data with controlled statistics
    
    """
            
    data_creator = SingleDataCreater(name=dataset_name, 
                                     number_of_data =600)
    
    
    for added_column in added_columns:
    
        data_creator.add_collumn(from_column_name= data_creator.label_name,
                                 new_column_name=added_column[0],
                                 distribution=added_column[1], 
                                 correlation= added_column[2],
                                 df= added_column[3]
                                 )
    
    DATA = data_creator.get_data()

    input_columns = [c for c in  list(DATA.columns) if not "label" in c]
    
    return  DATA[input_columns], DATA["label"]
        

variable_types             = ["controll_random","controll_constant","fluid"]
distributions              = ["normal","random","chisquare"]
dfs                        = [0,1,5,10,15]
fluid_correlations         = [0.35, 0.55, 0.75] + [-0.35,-0.55,-0.75]

const_correlations         = [0.7]
with_const_correlation     = [1,0]
dataset_name                = f"synthetic_single"

analyse_name                = "synthetic_multi_column_data"

label_index       = 0

step              = 0
seeds             = 100
SEED              = 7
shapley_steps     = 5

stability_index_counter = 0

shap_sample_size        = 30
model_counts            = 3

wb_result_path = "result/synth_data/multi_column/"
create_dir(wb_result_path)


all_emb_result_df = pd.DataFrame()
all_result_df = pd.DataFrame()
all_emb_result_df = pd.DataFrame()


how_many_step = seeds * len(dfs)* shapley_steps  * model_counts * len(fluid_correlations) * len(with_const_correlation)


standard_features      = 6
standard_n_informative = 4


from sklearn import datasets

fig = plt.figure(figsize=(8, 6))


flips                       = [round(std*0.01,2) for std in range(1,100,20)]


num                         = 700

multipliers                 = [1, 0.75, 0.5, 0.25, 0.1]
data_sizes                  = [std for std in range(num,50,-200)][::-1]
n_clusters_per_class        = 3


shapley_values_dict  = {}

#n_clusters_per_class = 3

all_shap_results_df = pd.DataFrame()
n_clusters_per_class_es =[std for std in range(2,5)]

all_number_counts = len(data_sizes) * len(flips) * len(n_clusters_per_class_es) * 3  * seeds

counter = 0
data_index = 0
number_of_data  = 100000

how_many_step = seeds * len(dfs)* shapley_steps  * model_counts * len(fluid_correlations) * len(with_const_correlation)


for seed in range(seeds):

    for data_size in data_sizes:
        
        for flip in flips:
    
            #seed = np.random.randint(10000)
            print(seed)
            
            X_g, y_g = datasets.make_classification(n_samples=number_of_data,
                n_features=standard_features,
                n_informative=standard_n_informative,
                n_redundant=0,
                n_repeated=0,
                n_classes=2,
                n_clusters_per_class=n_clusters_per_class,
                weights=None,
                flip_y=flip,
                class_sep=1.0,
                hypercube=True,
                shift=0.0,
                scale=1.0,
                shuffle=True,
                random_state=seed)
            

            for multiplier in multipliers:
                
                print(f"multiplier {multiplier}")
                
                shap_results_df = pd.DataFrame()
                shapley_values_dict= {}
                

                X_gen, y_gen = make_imbalance(
                            X_g,
                            y_g,
                            sampling_strategy=ratio_func,
                            **{"multiplier": multiplier, "minority_class": 1},
                        )
                
                
                feature_names = ['Ft %i' % i for i in range(standard_features)]
                
                reg_df = pd.DataFrame(X_gen, columns=feature_names)
                reg_df['y'] = y_gen
                corr_matrix = reg_df.corr().round(2)
            
                reg_df_sample   = reg_df.sample(data_size)
                
                X_data  = reg_df_sample[feature_names]
                Y_data  = reg_df_sample['y']
                         
                ratio = Counter(Y_data)
                porportion = get_ratio(ratio[1], ratio[0])
                
                
                print(porportion)
                
                cv_outer   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
                ci = 0
                

        
                for train_index, test_index in cv_outer.split(X_data, Y_data):
                    
                    all_full_models_df   = pd.DataFrame()                    
                    all_ci_models_df   = pd.DataFrame()

                    
                    percent_complete = round((step / how_many_step)  * 100, 2)
                    
                    os.system("clear")
                    print(f"seed: {seed}")
      
                    #percent_complete = round((step / how_many_step)  * 100, 2)
                    print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
     
                    ci+=1
        
                    X_train_full_hot, X_test_full_hot = X_data.iloc[train_index, :], X_data.iloc[test_index, :]
                    y_train, y_test = pd.DataFrame(Y_data).iloc[train_index,0], pd.DataFrame(Y_data).iloc[test_index,0]   
                    
                    all_feature_names = list(X_train_full_hot.columns)
                    
                    data_size = len(X_train_full_hot)
                    
    
                    #tree model #######################################################
                    
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
                    if (tp + fn):
                        TPR = tp / (tp + fn)
                    else:
                        TPR = 0
                        
                    #specificity
                    if (tn + fp):
                        TNR = tn / (tn + fp)
                    else:
                        TNR = 0
                    
    
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
                            
                            
                            # linear_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                            #                                         all_linear_shap_values.values)
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                     feature_names = all_feature_names)
                            
                        
                            # kernel_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                            #                                         all_kernel_shap_values[1])
    
    
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
                        if (tp + fn):
                            TPR = tp / (tp + fn)
                        else:
                            TPR = 0
                            
                        #specificity
                        if (tn + fp):
                            TNR = tn / (tn + fp)
                        else:
                            TNR = 0
                            
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
                                
                            
                                # kernel_shap_values, feature_names = collect_hot_encoded_features(shap_sample_data_test,
                                #                                         all_kernel_shap_values[1])
                                
                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                
                                all_tree_shap_values                    = TreeExplainer.shap_values(shap_sample_data_test)
                            
                
                                # tree_shap_values, feature_names     = collect_hot_encoded_features(shap_sample_data_test,
                                #                                         all_tree_shap_values[1])
                                
        
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
                        if (tp + fn):
                            TPR = tp / (tp + fn)
                        else:
                            TPR = 0
                            
                        #specificity
                        if (tn + fp):
                            TNR = tn / (tn + fp)
                        else:
                            TNR = 0
                            
                            
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
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                          feature_names = all_feature_names)
                        
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
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
        
              
                        all_full_models_df  = pd.concat([all_full_models_df,all_ci_models_df])     
            
                        all_full_models_df["flip"]                       = flip
            
                        all_full_models_df["multiplier"]                 = multiplier
                        all_full_models_df["shap_sample_size"]           = shap_sample_size
                    
                        all_full_models_df["train_data_size"]            = data_size
                        
                        all_full_models_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                        all_full_models_df["stability_index_counter"]    = stability_index_counter
                        all_full_models_df["it_seed"]                    = seed
            
            
                        all_emb_result_df = pd.concat([all_emb_result_df,all_full_models_df])
                        all_emb_result_df.to_excel(wb_result_path+f"part_{analyse_name}_result.xlsx")
                        
                         
                    all_emb_result_df.to_excel(wb_result_path+f"all_{analyse_name}_result.xlsx")
                          
                    try:   
                        all_emb_result_df.to_excel(wb_result_path+f"all_{analyse_name}_cp.xlsx")
                    except:
                        pass
                
                

                
                    
          