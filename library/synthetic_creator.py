#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:46:55 2022

@author: temuuleu
"""

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
import time

from datetime import datetime
# test classification dataset
from sklearn.datasets import make_classification
import seaborn as sn

#import pydbgen
import random
import scipy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter

import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import re
import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
#import plotly.express as px
#import plotly.figure_factory as ff


import numpy as np

# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt


# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#from ipywidgets import HBox, VBox
from scipy.stats import t
from scipy.stats import norm
from math import inf

import bs4 as bs
import requests


from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import beta
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import shap
shap.initjs()
from scipy.stats import chisquare

##dist_list = ['uniform','normal','exponential','lognormal','chisquare','beta']
import sklearn.datasets as dt
from scipy import linalg
import scipy.sparse as sp
# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numbers
import math
from fractions import Fraction

import math 

from library.learning_lib import create_dir
from library.learning_lib import *


class SingleDataCreater():

    def __init__(self, name="", 
                 number_of_data =100,
                 prob  = 0.5,
                 #data_path = "../data/complette_synthethic",
                 try_label = 1000,
                 label_index = 1):
        
      self.maxiteration               = try_label
      #self.data_path                  = data_path
      self.name                       = name
      self.label_name                 = "label"
      self.columns_created            = False
      self.columns_number             = 1
      self.columns                    = []
      self.datasets                   = {}
      
      self.labels                     = []
      self.number_of_data             = number_of_data
      self.parameter_names            = []
      self.parameter_distribution     = []
      self.parameter_correlations     = []
      self.dataframe                  = pd.DataFrame()
      
      self.ids                        = np.array([i_d for i_d in range(1,number_of_data+1)])
      self.dataframe["id"]            = self.ids
      
      
      
      
      """create labels until the given ratio ist reached"""
      #labels, label_ratio  = create_labels(number_of_data = number_of_data, ratio = ratio, max_iter=try_label)
      
      labels                              =  np.random.choice([0, 1], 
                                                  size=number_of_data, 
                                                  p=[round(1 -prob,2) ,prob])
      
      
      
      self.numerical_data_frame              = pd.DataFrame()
      #self.numerical_data_frame["label"]    = labels
      
      self.labels                           = labels
      #check for real label ratio
      #self.label_ratio                      = label_ratio
      #save the  label array to dataframe
      self.dataframe[self.label_name]       = self.labels 
      
      self.columns                          = []
 
      self.label_index                      = label_index
      self.dataframe["label_index"]         = label_index
      
      #number of dataset
      self.number_of_data                   = len(self.labels)
      #time to create the dataset
      self.all_elapsed_time                 = 0
      db_file_name                          = "data.xlsx"
      
      #create data path fall all db
      #self.db_path_all                      = os.path.join(self.data_path, str(int(label_ratio*100)))
      #create_dir(self.db_path_all)
      #all file path
      #self.db_path_all_excel                = os.path.join(self.db_path_all, db_file_name)
      #self.db_path_ratio                    = os.path.join(self.db_path_all,self.label_name +"_" +str(label_index))
      
      #self.db_path_ratio                    = self.db_path_all
      #create_dir(self.db_path_ratio)
      #ratio database file path
      #self.db_path_ratio_excel              = os.path.join(self.db_path_ratio,db_file_name)

      #all header file
      #header_file_name                      = "header.xlsx" 
      #self.header_file_path_all_excel       = os.path.join(self.db_path_all, header_file_name)
      #part header file
      #self.header_file_path_ratio_excel     = os.path.join(self.db_path_ratio, header_file_name)
      #plot path
      #self.plot_path                        = os.path.join(self.db_path_ratio , "plots")
      #create_dir(self.plot_path)
      
      #create_dir(self.db_path)
      #self.db_file_path                     = os.path.join(self.db_path,self.db_file)
      


      #self.dataframe.to_excel(self.db_path_ratio_excel,index=False)
      
      #self.header_label_df                   = pd.DataFrame( index=(0,))  
      #self.header_label_df["label_name"]     = self.label_name 
      #self.header_label_df["label_ratio"]    = self.label_ratio 
      #self.header_label_df["number_of_data"] = self.number_of_data 
      #self.header_label_df.to_excel(self.header_file_path_ratio_excel,index=False)
      
      # self.header_param_df                   = pd.DataFrame() 
      
      
      #all
      #all header
      # self.all_header_df                     = pd.DataFrame() 
      # self.all_header_df                     = pd.concat([ self.all_header_df,
      #                                                   self.header_label_df],
      #                                                  axis = 0)
      # self.all_header_df.to_excel(self.header_file_path_all_excel,index=False)
      
      #all data
      self.all_data_df                       = pd.DataFrame() 
      self.all_data_df           = pd.concat([self.all_data_df , pd.DataFrame(self.labels , columns=["label"])], axis=1)
      
      #self.dataframe["label_name"]           = self.label_name 

    def plot_label_hist(self):
        """print the distribution of label"""
        self.dataframe[self.label_name].hist() 
        plt.title("label : "+self.label_name)
        plt.show()
        
    def plot_input_hist(self,parameter_name):
        """print the distribution of label"""
        
        self.dataframe[parameter_name].hist() 
        plt.title(parameter_name)
        plt.show()
        
    def show_parameters(self):
        print(f"This function shows the save parameters:")
        print(f"{self.name}")
        print(f"Labelname       :   {self.label_name}")
        print(f"Size            :   {self.number_of_data}")
        print(f"Label Ratio     :   {self.label_ratio}")
        print(f"Parameters      :   {self.parameter_names}")
        print(f"Distribution    :   {self.parameter_distribution}")
        print(f"Correlation     :   {self.parameter_correlations}")
        print(f"Column is created:  {self.columns_created}")
        print(f"Time created in     {self.all_elapsed_time} Seconds ")


    def show_parameters(self):
        
        print("Parameters: ")
        print(f"Parameters:               {self.label_ratio} ")
        print(f"number_of_data:           {self.number_of_data} ")
        print(f"parameter_names:          {self.parameter_names} ")
        print(f"parameter_distribution:   {self.parameter_distribution} ")
        print(f"parameter_correlations:   {self.parameter_correlations} ")
        
        
    def show_covariance(self):
        if self.columns_created:
            self.dataframe[self.parameter_names].hist() 
            plt.show()

            corr = self.dataframe.corr()
            sns.heatmap(corr)
            plt.show()
            
            print("correlation matrix")


    def get_data(self):
        
        return self.all_data_df
    

    def add_collumn(self,
                    new_column_name="",
                    distribution="normal",
                    correlation=0, 
                    corr_type="pearsonr",
                    data_type="",
                    var_type="",
                    from_column_name="",
                    std=10, 
                    expected_value = 100, 
                    r = 2,
                    df = 0,
                    target_p = 0.05,
                    maxiteration = 1000,
                    number_of_categories  = 0,
                    categorical_even   = True
                    ):    
        
        """add new columns to the dataset"""
        if correlation == 0:
            corr_type="nocorrelation"

            
        if  not from_column_name in  self.dataframe.columns:
            print("the source columns doesn't exist")
            return 0
        
        
        col_name              = str(self.columns_number)+ "_"+new_column_name
        self.columns.append(col_name)
        

        data, elapsed_time,correlation,found_p,numerical_data_frame,found_min   = create_data_from_column(
                                  self.numerical_data_frame,
                                  self.dataframe,
                                  columnname = "label",
                                  correlation = correlation, 
                                  corr_type = corr_type, 
                                  data_type = data_type,
                                  distribution = distribution,
                                  df = df,
                                  target_p = target_p,
                                  maxiteration=maxiteration,
                                  number_of_categories = number_of_categories,
                                  categorical_even = categorical_even
                                  )
        
        
        
        
        self.numerical_data_frame = numerical_data_frame
        
        header_param__temp_df                           = pd.DataFrame( index=(0,))  
        header_param__temp_df["label_index"]            = self.label_index
        header_param__temp_df["parameter_names"]        = col_name
        header_param__temp_df["distribution"]           = distribution

        header_param__temp_df["data_type"]              = data_type
        header_param__temp_df["corr_type"]              = corr_type
        header_param__temp_df["var_type"]               = var_type

        header_param__temp_df["maxiteration"]           = maxiteration
        header_param__temp_df["elapsed_time"]           = elapsed_time
        now = datetime.now()
        header_param__temp_df["current_time"]           = now.strftime("%H:%M:%S")

        #if data_type == "numerical":
   
        data_values                                     = std * data + expected_value
        data_values                                     = np.round(data_values,2)
        standard                                        = np.round(np.std(data_values))
        expected_value                                  = np.round(np.mean(data_values))

        
        header_param__temp_df["p"]                      = round(found_p,2)
        header_param__temp_df["correlation"]            = round(correlation,2)
        header_param__temp_df["std"]                    = std
        
        header_param__temp_df["categorical_even"]       = ''
        header_param__temp_df["found_min"]              = ''
        header_param__temp_df["number_of_categories"]   = ''
            

        # else:
        #     standard                                        = 0
        #     expected_value                                  = 0
        #     header_param__temp_df["std"]                    = 0
            
        #     corr_type                                       = "chisquared"
        #     data_values                                     = data
        #     header_param__temp_df["p"]                      = round(found_p,2)
        #     header_param__temp_df["correlation"]            = ''
            
        #     header_param__temp_df["categorical_even"]       = categorical_even
        #     header_param__temp_df["found_min"]              = found_min
        #     header_param__temp_df["number_of_categories"]   = number_of_categories

                    
        print("created")
        print(" ")
        #document creation.

        # column_dict = {   
        #     "type"            : corr_type,
        #     "correlation"     : correlation,
        #     "distribution"    : distribution,
        #     "column_number"   : self.columns_number,
        #     "col_name"        : col_name,
        #     "standard"        : standard,
        #     "mean"            : expected_value
        #     }
        
        # self.parameter_names.append(col_name)    
        # self.parameter_distribution.append(distribution)    
        # self.parameter_correlations.append(correlation)
        # self.parameter_correlations.append(std)
        # self.columns.append(column_dict) 
        #self.dataframe[col_name]                     =  data_values
        
        
        #self.all_data_df                       = pd.DataFrame() 
        
        all_data_df = self.all_data_df
        
        
        data_values_df = pd.DataFrame(data_values , columns=[col_name])
        
        self.all_data_df = pd.concat([all_data_df,data_values_df],axis=1)
        
        # self.columns_number                          +=1
        # self.columns_created                         = True
        # self.all_elapsed_time                        +=elapsed_time
        # self.dataframe.to_excel(self.db_path_ratio_excel,index=False)
        
        
        # if self.label_name == from_column_name:
        #     from_column_type = "label"
        # else:
        #     from_column_type = "column"
            
        # #parameter type
        # header_param__temp_df["from_column_type"]    = from_column_type
        # header_param__temp_df["df"]                  = df

        # self.header_param_df                         = pd.concat([ self.header_param_df, 
        #                                                          header_param__temp_df], axis = 0)
        

        # self.header_df                               = pd.concat([self.header_label_df,
        #                                                           self.header_param_df ], axis = 1)
        
        
        
        # self.all_header_df = pd.concat([  self.all_header_df,self.header_df ], axis = 1)
        
        #self.header_df.to_excel(self.header_file_path_ratio_excel,index=False)
        
        # print(f"header_param_df      {self.header_param_df.shape}")
        # print(f"dataframe            {self.dataframe.shape}")
        # print(f"all_elapsed_time     {round(self.all_elapsed_time)}")   

