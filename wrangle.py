# Imports
import pandas as pd
import requests
import math
import os

print(f"Imports Loaded Successfully")

#----------------------------------------------------------------- ACQUIRE ----------------------------------------------------------------------------------

def get_data(filename):
    """This function loads a CSV file from local directory
    ---
    Format: df = function()
    """
    if os.path.isfile(f"{filename}.csv"):
        df = pd.read_csv(f"{filename}.csv", index_col=0)
        print(f"CSV File Found, Loading...")
    else:
        print(f"CSV File Not Found")

    return df

#----------------------------------------------------------------- PREPARE ----------------------------------------------------------------------------------
    
def prep_invoice(df):
    """This function takes in the df from 'income profit summary' CSV and:
        - dropped first two customers with invoices of -1 and 0
        - created new column off of index 'invoice'
        - changes date to datetime
        - drops percent profit and amount profit columns
        - runs for loop to change remaining columns to float and round(2)
        - return clean df
    ---
    Format: df = function()
    """  
    print(f"DataFrame acquired, cleaning...\n")
    # dropped two first customers with index [-1,0]
    df_1 = df
    df = df.drop(index=[-1,0])
    print(f"Dropped Negative Index Rows")
    
    # changes date column to datetime
    df.date = pd.to_datetime(df.date)
    print(f"Changed date to datetime type")
    
    # runs for loop to change dytpes and round to 2 places
    for col in df.iloc[:,2:]:
        df[col] = df[col].str.replace(',','')
        df[col] = df[col].astype(float).round(2)
    print(f"Changed numeric columns to floats, round to 2")
    
    # create new columns and delete old ones
    df['profit_per_part'] = df.parts_sale - df.parts_cost
    df['profit_per_labor'] = df.labor_sale - df.labor_cost
    df['profit'] = df.sale_total - df.total_cost
    print(f"Feature Engineereed Columns: Profit Per Part, Profit Per Labor, Profit")
    
    # binning to create small, medium, high ro orders
    bins = [ 0, 150, 750, 2000]
    labels = ['small', 'medium', 'large']
    df['profit_size'] = pd.cut(df['total_cost'], bins=bins, labels=labels, include_lowest=True)
    print(f"Binned To Create Order Sizes: Small, Medium, Large - {bins}")
    
    # drop derivative columns
    df = df.drop(columns={'sublet_cost','sublet_sale',\
                         'sale_total','total_cost','percent_profit','amount_profit'})
    print(f"Dropped Sublet Columns, Derived Columns")
    
    # drop NaN's
    df = df.dropna()
    df = df[df['labor_cost'] > 1]
    df= df[df['profit'] > 0]
    print(f"Dropped NaN's and all 0's from labor cost")
    
    # removing specific outliers
    for col in df.iloc[:,2:5]:
        df_clean = df[(df[col] < 2000) & (df[col] > 0)]
    print(f"\nOutliers Removed: Percent Original Data Remaining: {round(df_clean.shape[0]/df_1.shape[0]*100,0)}\n")
        
    return df_clean

# ------------------------------------------------------- Univariate Exploration --------------------------------------------------------------------------------

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
        
