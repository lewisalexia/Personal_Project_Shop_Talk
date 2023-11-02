import stats_conclude as sc

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import  mean_squared_error
from math import sqrt

from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------- UNIVARIATE EXPLORATION --------------------------------------------------------------------------------

def univariate_hist(df):
    """This function takes in a df and returns Seaborn histplots on all columns"""
    plt.figure(figsize=(12,6))
   
    for i, col in enumerate(df.columns):
        plt.tight_layout(pad=3.0)
        plot_number = i + 1
        plt.subplot(6,2, plot_number)
        sns.histplot(df[col], bins=20)
        plt.title(f"{col.replace('_',' ')}") 
        plt.xticks(rotation=45)

    plt.subplots_adjust(left=0.1,
                bottom=0, 
                right=0.9, 
                top=3, 
                wspace=0.6, 
                hspace=0.6)
    plt.show()
    
def univariate_desc(df):
    """This function takes in a df and returns descriptive analysis on numerical columns"""
    for col in df.columns:
        print(df[col].describe())
        print(f"\n") 

#-------------------------------------------------------------  SPLIT ----------------------------------------------------------------------------------
        
def split_classification(df, target):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.
    ---
    Format: train, validate, test = function()
    '''
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=.25,
                                       random_state=123, stratify=train_validate[target])
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test 

def split_regression(df):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.
    ---
    Format: train, validate, test = function()
    '''
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25,
                                       random_state=123)
    
    train.drop(columns={'date','customer','profit_size'}, inplace=True)
    validate.drop(columns={'date','customer','profit_size'}, inplace=True)
    test.drop(columns={'date','customer','profit_size'}, inplace=True)
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

#-------------------------------------------------------------  ENCODE ----------------------------------------------------------------------------------

def encode_classification(df):
    """This function encodes the categorical column for project shop talk
    and reorders the columns.
    ---
    Format: df = function()
    """
    dummy_df = pd.get_dummies(df[['profit_size']], dummy_na=False, drop_first=True)
    df_encoded = pd.concat([dummy_df, df], axis=1)

    # reorders columns
    df_encoded = df_encoded[[
                         'date',
                         'customer',
                         'parts_cost',
                         'labor_cost',
                         'parts_sale',
                         'labor_sale',
                         'profit_per_part',
                         'profit_per_labor',
                         'profit',
                         'profit_size',
                         'profit_size_medium',
                         'profit_size_large']]
    
    return df_encoded

#-------------------------------------------------------------  SCALE ----------------------------------------------------------------------------------

def minmax_scaler(train_model, validate_model, test_model):
    """This functions takes in the train, validate, test df's, creates the minmax scaler, fits it to train,
    uses it on train, validate, and test df's. Returns two graphs, one of the 
    original and one of the scaled data.
    ---
    Format: train_model_scaled, validate_model_scaled, test_model_scaled = function()
    """
    scaler = MinMaxScaler()
    
    # fit
    scaler.fit(train_model)
    
    # use
    train_model_scaled = scaler.transform(train_model)
    validate_model_scaled = scaler.transform(validate_model)
    test_model_scaled = scaler.transform(test_model)

    # viz
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(train_model, ec='black')
    plt.title('Original Data')
    plt.subplot(122)
    plt.hist(train_model_scaled, ec='black')
    plt.title('Scaled Data');
    
    # returns array - make into DF
    train_model_scaled = pd.DataFrame(train_model_scaled, columns=train_model.columns)
    validate_model_scaled = pd.DataFrame(validate_model_scaled, columns=validate_model.columns)
    test_model_scaled = pd.DataFrame(test_model_scaled, columns=test_model.columns)
    
    return train_model_scaled, validate_model_scaled, test_model_scaled
    
#-------------------------------------------------------------  QUESTIONS ----------------------------------------------------------------------------------    
     
def plot_1(df, col1, col2):
    """This function plots question 1 viz"""
    sns.lmplot(data=df, x=col1, y=col2, sharex=True, sharey=True, size=9, line_kws={'color':'purple'})
    plt.axhline(df[col2].mean(), color='black', linestyle=':', label='Avg Profit')
    plt.axvline(df[col1].mean(), color='red', linestyle=':', label='Avg Labor Cost')
    plt.title('Labor Cost Against Profit')
    plt.legend();
    
def plot_2(df, col1, col2, col3):
    """This function plots question 2 viz"""
    sns.lmplot(data=df, x=col1, y=col2, hue=col3, sharex=True, sharey=True, size=9, line_kws={'color':'purple'}, col=col3\
          ,col_wrap=2)
    plt.axhline(df[col2].mean(), color='black', linestyle=':', label='Avg Profit')
    plt.axvline(df[col1].mean(), color='red', linestyle=':', label='Avg Labor Cost')
    plt.title('Labor Cost Against Profit')
    plt.legend();
    
def plot_3(df, col1, col2):
    """This function plots question 3 viz"""
    sns.lmplot(data=df, x=col1, y=col2, sharex=True, sharey=True, size=9, line_kws={'color':'purple'})
    plt.axvline(df[col1].mean(), color='black', linestyle=':', label='Avg Parts Cost')
    plt.axhline(df[col2].mean(), color='red', linestyle=':', label='Avg Profit')
    plt.title('Part Cost Against Profit')
    plt.legend();
    
def plot_4(df, col1, col2, col3):
    """This function plots question 4 viz"""
    sns.lmplot(data=df, x=col1, y=col2, hue=col3, sharex=True, sharey=True, size=9, line_kws={'color':'purple'}, col=col3\
          ,col_wrap=2)
    plt.axhline(df[col2].mean(), color='black', linestyle=':', label='Avg Profit')
    plt.axvline(df[col1].mean(), color='red', linestyle=':', label='Avg Parts Cost')
    plt.title('Parts Cost Against Profit')
    plt.legend();
    
#-------------------------------------------------------------  X_train, etc... ----------------------------------------------------------------------------------    
    
def assign_variables(train, validate, test, target):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    ---
    Format = X_train, X_validate, X_test, y_train, y_validate, y_test = function()
    """
    # X_train, y_train, X_validate, y_validate, X_test, and y_test to be used for feature importance/modeling
    variable_list = []
    X_train = train.drop(columns={'date','customer','profit_per_part','profit_per_labor','profit'})
    variable_list.append(X_train)
    X_validate = validate.drop(columns={'date','customer','profit_per_part','profit_per_labor','profit'})
    variable_list.append(X_validate)
    X_test = test.drop(columns={'date','customer','profit_per_part','profit_per_labor','profit'})
    variable_list.append(X_test)
    y_train = train[target]
    variable_list.append(y_train)
    y_validate = validate[target]
    variable_list.append(y_validate)
    y_test = test[target]
    variable_list.append(y_test)

    return variable_list

def assign_reg_variables(train, validate, test, target):
    """
    """
    # X_train, y_train, X_validate, y_validate, X_test, and y_test to be used for feature importance/modeling
    variable_list = []
    X_train = train.drop(columns={'profit_per_part','profit_per_labor', 'profit'})
    variable_list.append(X_train)
    X_validate = validate.drop(columns={'profit_per_part','profit_per_labor', 'profit'})
    variable_list.append(X_validate)
    X_test = test.drop(columns={'profit_per_part','profit_per_labor', 'profit'})
    variable_list.append(X_test)
    y_train = train[target]
    variable_list.append(y_train)
    y_validate = validate[target]
    variable_list.append(y_validate)
    y_test = test[target]
    variable_list.append(y_test)

    return variable_list

#----------------------------------------------------------------- VIZZES ----------------------------------------------------------------------------------

def plot_heatmap(df):
    """This functions returns a heatmap of a df"""
    worth_corr = df.corr(method='spearman')
    sns.heatmap(worth_corr, cmap='PRGn', annot=True, mask=np.triu(worth_corr))
    plt.title(f"Correlation Heatmap")
    plt.show()
    
def univariate_hist(df):
    """This function takes in a df and returns Seaborn histplots on all columns"""
    plt.figure(figsize=(12,6))
   
    for i, col in enumerate(df.columns):
        plt.tight_layout(pad=3.0)
        plot_number = i + 1
        plt.subplot(4, 3, plot_number)
        sns.histplot(df[col])
        plt.title(f"{col.replace('_',' ')}") 
        plt.xticks(rotation=45)

    plt.subplots_adjust(left=0.1,
                bottom=0, 
                right=0.9, 
                top=1.5, 
                wspace=0.6, 
                hspace=0.6)
    plt.show()