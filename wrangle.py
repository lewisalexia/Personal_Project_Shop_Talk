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
    print(f"DataFrame acquired, cleaning...")
    # dropped two first customers with index [-1,0]
    df = df.drop(index=[-1,0])
    df = df.reset_index(drop=True)
    
    # changes date column to datetime
    df.date = pd.to_datetime(df.date)
    print(f"Changed date to datetime type")
    
    # drops derived and unneccessary columns
    df = df.drop(columns={'percent_profit','amount_profit'})
    print(f"Dropped percent and amount profit columns")
    
    # runs for loop to change dytpes and round to 2 places
    for col in df.iloc[:,2:]:
        df[col] = df[col].str.replace(',','')
        df[col] = df[col].astype(float).round(2)
    df['invoice'] = df.index.astype(str)
    print(f"Changed remining columns to floats, round to 2")
    print(f"DateFrame cleaned and ready for exploration")
        
    return df
        

def prep_reciept(df):
    """This function takes in the df from 'receipt by payment type' CSV and:
        - created new column off of index 'invoice'
        - changed date to datetime
        - fill NaN in cust_id with 1 to represent counter sales
        - changed cust_id to int from float
        - fill index NaN value with delete
            * dropped index rows: Late Fee, On Account, and delete
        - return clean df
    ---
    Format: df = function()
    """
    # changed date to datetime
    df.date = pd.to_datetime(df.date)
    print(f"Changed date to datetime type")
    
    # fill NaN cust_id with 1 as they are counter sales
    df.cust_id = df.cust_id.fillna(1)
    print(f"Filled NaN's in cust_id with 1 = 'Counter Sale'")
    
    # changed cust_id to int instead of float
    df.cust_id = df.cust_id.astype(int)
    
    # fill index na with value
    df.index = df.index.fillna('delete')
    
    # drop index values
    df = df.drop(index=['Late Fee', 'On Account', 'delete'])
    print(f"Dropped Index items containing Late Fee, On Account and delete: 3 rows")
    
    # created new column off the index (used to concat later)
    df['invoice'] = df.index
    print(f"DateFrame cleaned and ready for exploration")

    return df

def prep_income(df):
    """This function takes in the df from 'income distribution' CSV and:
        - dropped first two customers with invoices of -1 and 0
        - changes date to datetime
        - drops prct_tech_part for having 12,062 nulls
        - replace misc writers and tech with Jeff Costa
        - lowercased all writers and techs
        - return clean df
    ---
    Format: df = function()
    """  
    # dropped two first customers with index [-1,0]
    df = df.drop(index=[-1,0])
    print(f"Adjusted index by dropping [-1,0]")
    
    # changes date column to datetime
    df.date = pd.to_datetime(df.date)
    print(f"Changed date to datetime type")
    
    # dropped column with 12,062 nulls
    df = df.drop(columns='prct_tech_part')
    
    # lowered names
    df.writer = df.writer.str.lower()
    df.tech = df.tech.str.lower()
    
    # replace misc writers with jeff and clean other names
    df.writer = df.writer.replace({'<none>':'jeff', 'please select, service writer':'jeff','costa, jeff':'jeff', 'costa, gary':'gary',\
                                   'walters, jerrii':'jerrii','costa, pam':'pam','casto, bill':'bill','konen, phil':'phil','branson, todd':'todd'})
    df.tech = df.tech.replace({'please select, technician':'jeff','costa, jeff':'jeff','branson, todd':'todd','konen, phil':'phil','foehl, matt':'matt',\
                               'nofer, tim':'tim', 'eaton, matt':'eaton', 'oliver, nick':'nick'})
    
    # make prct_tech labor to rounded prct
    df.prct_tech_labor = df.prct_tech_labor.round(2)*100
    print(f"DateFrame cleaned and ready for exploration")
    
    return df

def prep_yearly(df):
    """This function takes in the df from 'yearly' CSV and:
        - changes sales_amount from object to float
        - changes index to datetime
        - returns clean df
    ---
    Format: df = function()
    """
    # changes sales_amount from object to float
    df.sales_amount = df.sales_amount.str.replace(',','').astype(float)
    
    # change index to datetime
    df.index = pd.to_datetime(df.index, format='%Y')
    print(f"DateFrame cleaned and ready for exploration")
    
    return df

