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
    
#-------------------------------------------------------------  DESCRIPTIVES AND SPLIT ----------------------------------------------------------------------------------

def univariate_desc(df):
    """This function takes in a df and returns descriptive analysis on numerical columns"""
    for col in df.columns:
        print(df[col].describe())
        print(f"\n")        
        
def outliers(df):
    """This function uses a built-in outlier function using IQR range and 1.5 multiplier
    to scientifically identify all outliers in the zillow dataset and then 
    print them out for each column.
    ---
    Format: upper_bound, lower_bound = function()
    """
    for col in df.columns:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        lower_bound = q1 - (1.5 * iqr)
        print(f"{col}: upper = {upper_bound}, lower = {lower_bound}") 

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
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test 

def remove_model_outliers(df):
    """This function takes in a df and removes the outliers using IQR and default multiplier of 1.5
    returning a clean df
    ---
    Format: df = function()
    """
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numerical varibles

    for col in df.columns:
        if col in df.select_dtypes(include=['int64', 'float64']):
            col_num.append(col)
        else:
            col_cat.append(col)

    for col in col_cat:
        print(f"{col.capitalize().replace('_',' ')} is excluded from this function.")
    print(f'-----------------------------------------')

    print(f'Outliers Calculated with IQR Ranges, multiplier 1.5')
    for col in col_num:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = q3 - q1
        upper_bound = q3 + (1.5 * iqr)
        lower_bound = q1 - (1.5 * iqr)
        print(f"{col.capitalize().replace('_', ' ')} between {lower_bound.round(2)} and {upper_bound.round(2)}") 
        df_clean = df[(df[col] < upper_bound) & (df[col] > lower_bound)] 
        
    print(f"\nOutliers Removed: Percent Original Data Remaining: {round(df_clean.shape[0]/df.shape[0]*100,0)}\n")
    return df_clean

def standard_scaler(train, validate, test):
    """This functions takes in the train, validate, test df's, creates the standard scaler, fits it to train,
    uses it on train, validate, and test df's. Returns two graphs, one of the 
    original and one of the scaled data.
    ---
    Format: train_std_scaled, validate_robust_scaled, test_std_scaled = function()
    """
    scaler = StandardScaler()
    # fit
    scaler.fit(train)
    # use
    train_std_scaled = scaler.transform(train)
    validate_std_scaled = scaler.transform(validate)
    test_std_scaled = scaler.transform(test)
    # viz
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(train_std_scaled, bins=25, ec='black')
    plt.title('Scaled');
    
    # returns array - make into DF
    train_std_scaled = pd.DataFrame(train_std_scaled, columns=train.columns)
    validate_std_scaled = pd.DataFrame(validate_std_scaled, columns=validate.columns)
    test_std_scaled = pd.DataFrame(test_std_scaled, columns=test.columns)
    
    return train_std_scaled, validate_std_scaled, test_std_scaled

def robust_scaler(train, validate, test):
    """This functions takes in the train, validate, test df's, creates the robust scaler, fits it to train,
    uses it on train, validate, and test df's. Returns two graphs, one of the 
    original and one of the scaled data."""
    scaler = RobustScaler()
    # fit
    scaler.fit(train)
    # use
    train_robust_scaled = scaler.transform(train)
    validate_robust_scaled = scaler.transform(validate)
    test_robust_scaled = scaler.transform(test)
    # viz
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(train_robust_scaled, bins=25, ec='black')
    plt.title('Scaled'); 
    
def assign_variables(train, validate, test, target):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    ---
    Format = X_train, X_validate, X_test, y_train, y_validate, y_test = function()
    """
    # X_train, y_train, X_validate, y_validate, X_test, and y_test to be used for feature importance/modeling
    variable_list = []
    X_train = train.drop(columns={'date','customer','invoice','profit',\
                                 'sale_total','total_cost'})
    variable_list.append(X_train)
    X_validate = validate.drop(columns={'date','customer','invoice','profit',\
                                 'sale_total','total_cost'})
    variable_list.append(X_validate)
    X_test = test.drop(columns={'date','customer','invoice','profit',\
                                 'sale_total','total_cost'})
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
    

    
    
def total_monthly(df):
    """This function displays monthly charges causing churn as they increase.
    """
    sns.scatterplot(data=df, x=df.monthly_charges, \
    y=df.total_charges, hue=df.churn)
    plt.axvline(x=df.monthly_charges.mean(), label='Average Monthly Charge', color='red', linewidth=2)
    plt.title('High Monthly Charges Creates Churn')
    plt.legend()
    plt.show()



def box_plot_monthly_multiple(df):
    """This function displays that mean monthly charges are higher for churned
    customers with and without multiple lines but ESPECIALLY for multiple line
    customers.
    """
    sns.boxplot(x=df.multiple_lines, \
                y=df.monthly_charges, hue=df.churn)
    plt.title('Houston, We Have a Problem...')
    plt.axhline(y=df.monthly_charges.mean(), label='Average Monthly Charge', color='red', linewidth=2)
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.legend([0, 1], ['No', 'Yes'])
    plt.show()


def phone_fiber(df):
    """This function displays that all fiber customers have phone service.
    """
    sns.barplot(data=df, x='phone_service', y='internet_service_type_Fiber optic')
    plt.title('All Fiber Customers Have Phone Service')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.ylabel('Fiber Internet')
    plt.xlabel('Phone Service')
    plt.show()

def monthly_phone_churn(df):
    """This function displays the churn rate of customers with one or more lines of phone
    service.
    """
    sns.barplot(data=df, x='churn', \
    y='contract_type', hue='multiple_lines')
    plt.axvline(x=df['churn'].mean(), label='Average Churn Rate', color='red', linewidth=2)
    plt.title('Phone Contracts Well Above Average Churn Rate')
    plt.legend()
    plt.show()


def fiber_average_cost(df):
    """This function displays churn customers being charged more than average monthly price.
    """
    sns.scatterplot(data=df, x='monthly_charges', y='total_charges', hue='phone_service')
    plt.axvline(x=df['monthly_charges'].mean(), label='Average Churn Rate', color='red', linewidth=2)
    plt.title('There Is Wiggle Room To Reduce Price')
    plt.show()

def monthly_contract(df):
    """This function displays the slope of charge for phone customers. The purpose is to
    show there is room to level the price.
    """
    sns.scatterplot(data=df, x='monthly_charges', y='total_charges', hue='contract_type')
    plt.title('Monthly Contracts Make Up Majority of Customer Base')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.show()