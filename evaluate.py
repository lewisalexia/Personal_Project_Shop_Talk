#Imports
#import warnings
import warnings
warnings.filterwarnings("ignore")

#import libraries
import pandas as pd
import numpy as np

#import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
from pydataset import data

#sklearn imports
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




# -- Feature Selection

def select_kbest(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    ---
    return: a df of the selected features from the SelectKBest process
    ---
    Format: kbest_results = function()
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    kbest_results = pd.DataFrame(
                dict(p_value=kbest.pvalues_, feature_score=kbest.scores_),
                index = X.columns)

    return kbest_results.sort_values(by=['feature_score'], ascending=False).head(k)

def kbest_to_df(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    ---
    return: a df of the selected features from the SelectKBest process
    ---
    Format: X_train_scaled_KBtransformed = function()
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    kbest_results = pd.DataFrame(
                dict(p_value=kbest.pvalues_, feature_score=kbest.scores_),
                index = X.columns)
    # we can apply this mask to the columns in our original dataframe
    X.columns[kbest.get_support()]
    
    # return df of features
    X_train_scaled_KBtransformed = pd.DataFrame(
        kbest.transform(X),
        columns = X.columns[kbest.get_support()],
        index=X.index)

    return X_train_scaled_KBtransformed.head(k)


def show_features_rankings(X_train_scaled, rfe):
    """
    Takes in a dataframe and a fit RFE object in order to output the rank of all features
    """
    # Dataframe of rankings
    ranks = pd.DataFrame({'rfe_ranking': rfe.ranking_}
                        ,index = X_train_scaled.columns)
    
    ranks = ranks.sort_values(by="rfe_ranking", ascending=True)
    
    return ranks

def rfe(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the recursive feature elimination process
        & a df of all rankings
    '''
    #MAKE the thing
    rfe = RFE(LinearRegression(), n_features_to_select=k)
    #FIT the thing
    rfe.fit(X, y)
        
    # use the thing
    features_to_use = X.columns[rfe.support_].tolist()
    
    # we need to send show_feature_rankings a trained/fit RFE object
    all_rankings = show_features_rankings(X, rfe)
    
    return all_rankings

# --- Regression

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

# ---- VIZZES

def plot_model_predictions(p1, p2, p3, y):
    """
    """
    plt.scatter(pred_lr2, y_train, label='linear regression')
    plt.scatter(pred_lars, y_train, label='LassoLars')
    plt.scatter(pred_glm, y_train, label='GLM')
    plt.plot(y_train, y_train, label='_nolegend_', color='grey')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81))

    plt.title("Where are predictions more extreme? More modest? Overfit?")
    plt.ylabel("Actual Profit")
    plt.xlabel("Predicted Profit")
    plt.legend();

def plot_model_residuals(p1, p2, p3, y):
    """
    """
    plt.axhline(label="No Error")

    plt.scatter(y_train, pred_lr2 - y_train, alpha=.5, color="red", label="LinearRegression")
    plt.scatter(y_train, pred_lars - y_train, alpha=.5, color="green", label="LassoLars")
    plt.scatter(y_train, pred_glm - y_train, alpha=.5, color="yellow", label="GLM")

    plt.legend()
    plt.title("Do the size of errors change as the actual value changes?")
    plt.xlabel("Actual Profit")
    plt.ylabel("Residual: Predicted Profit - Actual Profit");

def plot_model_actual_predicted(y, p1, p3):
    """
    """
    plt.hist(y_train, color='blue', alpha=.4, label="Actual")
    plt.hist(pred_lars, color='green', alpha=.9, label="LassoLars")
    plt.hist(pred_glm, color='orange', alpha=.7, label='GLM')

    plt.xlabel("Profit")
    plt.ylabel("Number of Customers")
    plt.title("Comparing the Distribution of Actual to Predicted Profit")
    plt.legend();