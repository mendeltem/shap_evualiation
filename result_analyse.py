#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:35:28 2022

@author: temuuleu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols




def compare_shap_glas(real_result_df):
    
    real_result_df["model_name"].unique()
    
    model_name = 'ExplainableBoostingClassifier'
    explainers = ["ExplainableBoosting_internal","kernel_explainer_shap"]
    
    concat_result = pd.DataFrame()
    
    
    p_temp    =  pd.DataFrame()
  
    pv_result = pd.DataFrame()
    
    pair_columns = []
    
    for explainer in explainers :
        
        s_column = []
    
        filter_for  = "score"
        
        model_result_df   = real_result_df[(real_result_df["model_name"] == model_name) ]
        
        explainer_1_df    = model_result_df[model_result_df["explainer"] == explainer]
        score_columns_1    = [sc for sc in  explainer_1_df.columns if filter_for in sc ]    
        score_columns_1    = [sc for sc in  score_columns_1 if not "&" in sc ]   
    
    
        filter_for  = "value"
        value_columns      = [sc for sc in  explainer_1_df.columns if filter_for in sc ]    
        value_columns    = [sc for sc in  value_columns if not "&" in sc ]  
        
 
        column_index = 0
        df_result = pd.DataFrame()
        
        
        for s_1 in score_columns_1:
            columns_name = explainer+s_1+"_score_values"
        
            s_column.append(columns_name)
            
            df = explainer_1_df[s_1].abs()
            normalized_df=(df-df.min())/(df.max()-df.min())
            explainer_1_df[s_1] = normalized_df
            
            pv_result[columns_name]     =  explainer_1_df[s_1]
            
        pair_columns.append(s_column)
            
        
        
    for explainer in explainers :
        
        s_column = []
    
        filter_for  = "score"
        
        model_result_df   = real_result_df[(real_result_df["model_name"] == model_name) ]
        
        explainer_1_df    = model_result_df[model_result_df["explainer"] == explainer]
        score_columns_1    = [sc for sc in  explainer_1_df.columns if filter_for in sc ]    
        score_columns_1    = [sc for sc in  score_columns_1 if not "&" in sc ]   
    
    
        filter_for  = "value"
        value_columns      = [sc for sc in  explainer_1_df.columns if filter_for in sc ]    
        value_columns    = [sc for sc in  value_columns if not "&" in sc ]  
        
 
        column_index = 0
        df_result = pd.DataFrame()
        
            
        for s_i,s_1 in enumerate(score_columns_1):
            
            values_1 = pv_result[pair_columns[0][s_i]]
            values_2 = pv_result[pair_columns[1][s_i]]
            
            tstatistic, pvalue = stats.ttest_ind(values_1, values_2)
            
            columns_ =[]
            df_temp = pd.DataFrame(index=(column_index,))
            explainer_1_df[s_1] = explainer_1_df[s_1].abs()
            
            df = explainer_1_df[s_1]
            normalized_df=(df-df.min())/(df.max()-df.min())
            explainer_1_df[s_1] = normalized_df
            

            cutlen = len(filter_for) +1
    
            df_temp["column_name"] = s_1[cutlen:]
            df_temp[explainer+"_mean_score_value"] =  explainer_1_df.groupby(["ci"]).mean()[s_1].mean().round(4)
            df_temp[explainer+"_std_score_value"]  =  explainer_1_df.groupby(["ci"]).mean()[s_1].std().round(4)
            
            columns_ += [explainer+"_mean_score_value"]
            columns_ += [explainer+"_std_score_value"]
            
            df_temp["p_value"] = round(pvalue,2)
            df_result = pd.concat([df_result ,df_temp ])
            
        concat_result = pd.concat([concat_result ,df_result ] , axis =1)
    
    return concat_result.T.drop_duplicates().T
        


# def create_multi_mean_table(balanced_data, columns_name ="data_sample_size" ,all_shap_name = 'kernelshap' ):

#     sample_sizes = list(balanced_data[columns_name].unique())
    
    
#             shapley_value_name = k[cutlen:]
#     cutlen = len(all_shap_name) +1
                
#     kernel_shap_columns        = [sc for sc in balanced_data.columns if all_shap_name in sc and sc in label_no_balanced.columns ]    
         
#     index = 0
#     df_result = pd.DataFrame()
    
#     for N in sample_sizes:
#         print(N)
    
#         data_n = balanced_data[balanced_data[columns_name] == N ]
        
#         for k in  kernel_shap_columns:
            
#             df_temp = pd.DataFrame(index=(index,))
#             shapley_value_name = k[cutlen:]
            
#             df_temp["shapley_value_name"]          =   shapley_value_name
            
#             df_temp["mean"]        =   np.round(np.mean( data_n[k]),2)
#             df_temp["std"]          =   np.round(np.std( data_n[k]),2)
#             df_temp["N"]              =   N
    
#             df_result = pd.concat([df_result ,df_temp ])
            
#             index += 1

#     return df_result
        
    
def ttest_b_fast_kernel(dataframe_df, name_1 = "fastshap" , name_2 = "kernelshap" ):

    df_result = pd.DataFrame()
         
    fast_shap_columns   = [sc for sc in  dataframe_df.columns if name_1 in sc ]    
    kernel_shap_columns = [sc for sc in  dataframe_df.columns if name_2 in sc ]    
    
    index = 0
    
    for f , k in zip(fast_shap_columns, kernel_shap_columns):
        
        df_temp = pd.DataFrame(index=(index,))
        
        shapley_value_name = k[11:]
        tstatistic, pvalue = stats.ttest_ind(dataframe_df[f], dataframe_df[k])
        
        df_temp["shapley_value"] = shapley_value_name
        df_temp["pvalue"]        =  np.round(pvalue,2)
        
        df_temp["mean"]          =  np.round(np.mean( dataframe_df[f]),2)
        df_temp["std"]           =  np.round(np.std( dataframe_df[f]),2)
        df_temp["len"]           = len(dataframe_df[f])
        
        #print(f"{shapley_value_name:>20}  pvalue : {round(pvalue,2) :>5}")
        
        df_result = pd.concat([df_result ,df_temp ])
        
        index+=1
    
    return df_result
        

def ttest_b_two_dataframe_shaply(label_balanced,label_no_balanced, all_shap_name        = "kernelshap" ):

    df_result = pd.DataFrame()
    
    cutlen = len(all_shap_name) +1
            
    kernel_shap_columns        = [sc for sc in label_balanced.columns if all_shap_name in sc and sc in label_no_balanced.columns ]    
     
    index = 0
    
    for k in  kernel_shap_columns:
        
        df_temp = pd.DataFrame(index=(index,))
        
        shapley_value_name = k[cutlen:]
        
        tstatistic, pvalue = stats.ttest_ind(label_balanced[k], label_no_balanced[k])
        
        df_temp["shapley_value"] = shapley_value_name
        df_temp["pvalue"]        =  np.round(pvalue,2)
        
        df_temp["mean_1"]          =  np.round(np.mean( label_balanced[k]),2)
        df_temp["std_1"]           =  np.round(np.std( label_balanced[k]),2)
        
        df_temp["mean_2"]          =  np.round(np.mean( label_no_balanced[k]),2)
        
        df_temp["std_2"]           =  np.round(np.std( label_no_balanced[k]),2)
        df_temp["len"]           = len(label_no_balanced[k])
        
        #print(f"{shapley_value_name:>20}  pvalue : {round(pvalue,2) :>5}")
        
        df_result = pd.concat([df_result ,df_temp ])
        
        index+=1
        
    return df_result




def ttest_two_dataframes(data_1, data_2,all_shap_name = 'kernelshap' ):

    df_result = pd.DataFrame()
        
    cutlen = len(all_shap_name) +1
                
    kernel_shap_columns        = [sc for sc in balanced_data.columns if all_shap_name in sc and sc in label_no_balanced.columns ]    
         
    index = 0
    
    
    for k in  kernel_shap_columns:
        
        df_temp = pd.DataFrame(index=(index,))
        shapley_value_name = k[cutlen:]
    
        tstatistic, pvalue = stats.ttest_ind(data_1[k], data_2[k])
        
        df_temp["shapley_value"] = shapley_value_name
        df_temp["pvalue"]        =  np.round(pvalue,2)
        
        df_temp["mean_1"]          =  np.round(np.mean( data_1[k]),2)
        df_temp["std_1"]           =  np.round(np.std( data_1[k]),2)
        
        df_temp["mean_2"]          =  np.round(np.mean( data_2[k]),2)
        
        df_temp["std_2"]           =  np.round(np.std( data_2[k]),2)
        df_temp["len"]           = len(data_2[k])
        
        #print(f"{shapley_value_name:>20}  pvalue : {round(pvalue,2) :>5}")
        
        df_result = pd.concat([df_result ,df_temp ])
        
        index+=1
        
    return df_result

    


real_result_path                  = "result/real_data/all_real_data_seeds_result_cp.xlsx"
synth_single_column_result_path   = "result/synth_data/single_column/all_synthetic_single_column_data_cp.xlsx"  
synth_multi_column_result_path    = "result/synth_data/multi_column/all_synthetic_multi_column_data_cp.xlsx"  

#loading the result
real_result_df                    = pd.read_excel(real_result_path, index_col=0)  
synth_single_column_result_df     = pd.read_excel(synth_single_column_result_path, index_col=0)  
synth_multi_column_result_df      = pd.read_excel(synth_multi_column_result_path, index_col=0)  
  


#controll columns for real data
general_compared =  compare_shap_glas(real_result_df)



general_compared.columns



colnames = list(general_compared["column_name"] )

mean_ex = general_compared['ExplainableBoosting_internal_mean_score_value']




general_compared.columns


single_column = colnames[0]
p_val =  general_compared[general_compared["column_name"]  == colnames[0]]['p_value'].values[0]

mean_glass =  general_compared[general_compared["column_name"]  == colnames[0]]['ExplainableBoosting_internal_mean_score_value'].values[0]
std_glassx =  general_compared[general_compared["column_name"]  == colnames[0]]['ExplainableBoosting_internal_std_score_value'].values[0]

mean_shap =  general_compared[general_compared["column_name"]  == colnames[0]]['kernel_explainer_shap_mean_score_value'].values[0]
std_shap =  general_compared[general_compared["column_name"]  == colnames[0]]['kernel_explainer_shap_std_score_value'].values[0]

# Create lists for the plot
materials = ['Glass_'+colnames[0], 'Shap_'++colnames[0]]
x_pos = np.arange(len(materials))
CTEs = [ mean_glass, mean_shap]
error = [ std_glassx, std_shap]


# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel(f'Difference Between internal Gals Box')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title(f'Difference Between internal Gals Box  P Value {p_val}')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()






ecolor='#ED2EC6'


font_size = 20

fig, axes = plt.subplots(1,1,
                         sharex='col',
                         sharey=True, 
                         figsize=(40, 16)
                        )

axes[0,0].errorbar(10,mean_glass,std_glassx, 
                   color="#0a7821", label='features_random_'+str(colnames[0]),ecolor=ecolor)






errorbar(N_c1_df_0['N'],N_c1_df_0['mean_features_0'],N_c1_df_0['std_features_0'], 
                       color="#0a7821", label='features_random_'+str(col_names[columns_list[0] ]),ecolor=ecolor)
    



plt.plot()



general_compared.drop(columns=["column_name"])


list(real_result_df.columns)

#glass box vs black box 

import scipy.stats as st


    
    

# #black box model

# ba_mean = real_result_df.groupby(["model_name",
#                                     "data_sample_size",
#                                     "age_sex_balance",
#                                     "label_balance"]).mean()["ba"].round(1).astype(str)


# ba_std = real_result_df.groupby(["model_name",
#                                 "data_sample_size",
#                                 "age_sex_balance",
#                                 "label_balance"]).std()["ba"].round(1).astype(str)



# ba = ba_mean+"\u00B1"+ba_std 


# #Balanced Accuracy

# model_name = "RandomForestClassifier"


# model = ols('ba ~ model_name', data = all_real_data_blackbox_result_df).fit()    
# aonva_result =   sm.stats.anova_lm(model, type=2)


# model_data     =  all_real_data_blackbox_result_df[(all_real_data_blackbox_result_df["model_name"] == model_name)]


# #label balance test
# label_balanced = model_data[(model_data["age_sex_balance"] == 1) \
#                                  &(model_data["label_balance"]   == 1) ] 
     
# label_no_balanced = model_data[(model_data["age_sex_balance"] == 1) \
#                                  &(model_data["label_balance"]   == 0)  ]
    
    
# #check for difference between kernel and other shapley values
# ttest_shap_balanced_label      =    ttest_b_fast_kernel(label_balanced)
# ttest_shap_not_balancedlabel   =    ttest_b_fast_kernel(label_no_balanced)

# #check if there is difference 
# ttest_shap_balanced           =   ttest_two_dataframes(label_balanced,label_no_balanced, all_shap_name = "kernelshap" )
# ttest_shap_balanced           =   ttest_two_dataframes(label_balanced,label_no_balanced, all_shap_name = "fastshap" )


# #gender age balance test
# gender_age_balanced = model_data[(model_data["age_sex_balance"] == 1) \
#                                  &(model_data["label_balance"]   == 1) ]
    
       
# gender_age_not_balanced = model_data[(model_data["age_sex_balance"] == 0) \
#                                  &(model_data["label_balance"]   == 1) ]
    
     
# #check for difference between kernel and other shapley values
# ttest_shap_balanced_gender              =    ttest_b_fast_kernel(gender_age_balanced)
# ttest_shap_not_balanced_gender          =    ttest_b_fast_kernel(gender_age_not_balanced)

# #check if there is difference 
# ttest_shap_kernelshap_gender     =   ttest_two_dataframes(gender_age_balanced,gender_age_not_balanced, all_shap_name = "kernelshap" )
# ttest_shap_fastshap_gender       =   ttest_two_dataframes(gender_age_balanced,gender_age_not_balanced, all_shap_name = "fastshap" )



# #filter only balanced data
# balanced_data  = model_data[((model_data["age_sex_balance"] == 1) &(model_data["label_balance"]   == 1)) ]

    
# model = ols('ba ~ data_sample_size', data = balanced_data).fit()    
# aonva_result =   sm.stats.anova_lm(model, type=2)



# #data_sizes
# balanced_data["data_sample_size"].unique()


# data_100 = balanced_data[(balanced_data["data_sample_size"] == 100) ]
# data_200 = balanced_data[(balanced_data["data_sample_size"] == 200) ]
# data_300 = balanced_data[(balanced_data["data_sample_size"] == 300) ]
# data_400 = balanced_data[(balanced_data["data_sample_size"] == 400) ]
# data_500 = balanced_data[(balanced_data["data_sample_size"] == 500) ]
# data_600 = balanced_data[(balanced_data["data_sample_size"] == 600) ]
    


# ttest_shap_kernelshap_100_200    =   ttest_two_dataframes(data_100,data_200, all_shap_name = "kernelshap" )
# ttest_shap_kernelshap_200_300    =   ttest_two_dataframes(data_200,data_300, all_shap_name = "kernelshap" )
# ttest_shap_kernelshap_300_400    =   ttest_two_dataframes(data_300,data_400, all_shap_name = "kernelshap" )
# ttest_shap_kernelshap_400_500    =   ttest_two_dataframes(data_400,data_500, all_shap_name = "kernelshap" )
# ttest_shap_kernelshap_500_600    =   ttest_two_dataframes(data_500,data_600, all_shap_name = "kernelshap" )



# multi_table  = create_multi_mean_table(balanced_data, columns_name ="data_sample_size" ,all_shap_name = 'kernelshap' )


# materials = list(multi_table["N"].unique())
# x_pos = np.arange(len(materials))

# names = list(multi_table["shapley_value_name"].unique())

# for name in names:

#     one_value_data  = multi_table[multi_table["shapley_value_name"] == name]

#     MEANS =  [ float(one_value_data[ one_value_data["N"] == n]["mean"].values) for n in materials]
#     ERRORS =  [ float(one_value_data[ one_value_data["N"] == n]["std"].values) for n in materials]
   
#     # Build the plot
#     fig, ax = plt.subplots()
#     ax.bar(x_pos, MEANS, yerr=ERRORS, align='center', alpha=0.5, ecolor='black', capsize=10)
#     ax.set_ylabel(f'Shapley Values')
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(materials)
#     ax.set_title(f'Shapley Values for {name}')
#     ax.yaxis.grid(True)
    
#     # Save the figure and show
#     plt.tight_layout()
#     plt.savefig(f'bar_plot_{name}_with_error_bars.png')
#     plt.show()
    
    











fig, axes = plt.subplots(2,2,
                          sharex='col',
                          sharey=True, 
                          figsize=(40, 16)
                        )

fig.suptitle("Shapley Values", size=40)





