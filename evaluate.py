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
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score



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

def baseline_classification(train_c):
    """This function creates a baseline prediction column on the train dataset used for classification
    and returns the baseline accuracy rating
    ---
    Format: baseline_accuracy = function()
    """
    # establish baseline accuracy
    train_c['baseline_prediction'] = 'small'
    baseline_accuracy = (train_c.baseline_prediction == train_c.profit_size).mean()
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}%')

    return baseline_accuracy

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

# ------ Classification

def all_4_classifiers(X_train, y_train, X_validate, y_validate, nn, baseline_accuracy):
    """This function takes in the train and validate datasets, a KNN number to go 
    to (exclusive) and returns models/visuals/explicit statments for decision tree, 
    random forest, knn, and logistic regression.

    Decision Tree:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
    
    Random Forest:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
        * visually presents feature importance

    KNN:
        * runs for loop to discover best fit "number of neighbors". nn argument
            is to set the end range for neighbors for loop (exclusive).
        * explicitly identifes the number of features sent in with column names
        * explicitly identifies the best fit number of neighbors
        * explicitly states accuracy scores for train, validate, and baseline
        * visually represents findings and identifies best fit neighbor size

    Logistic Regression:
        * random_seed = 123
        * runs logit on train and vaidate set
        * prints model, baseline accuracy, and a classification report
        
    Format: scores_df = functions()
    """
    # DECISION TREE
    print(f"DECISION TREE")
    scores_all=[]
    
    for i in range(1,3):
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        print(f"For depth of {i}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([i, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth_decision_tree', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc
        
        # print baseline
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}')
          
        # can plot to visulaize
    plt.figure(figsize=(6,6))
    plt.plot(scores_df.max_depth_decision_tree, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth_decision_tree, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.fill_between(scores_df.max_depth_decision_tree, scores_df.train_acc, scores_df.validate_acc, alpha=.4)
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,3, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
    # RANDOM FOREST
    print(f"RANDOM FOREST")
    scores_rf=[]

    for i in range(1,3):
        rf = RandomForestClassifier(random_state = 123,max_depth = i)
        rf.fit(X_train, y_train)
        train_acc_rf = rf.score(X_train, y_train)
        print(f"For depth of {i:2}, the accuracy is {round(train_acc_rf,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc_rf = rf.score(X_validate, y_validate)

        # append to rf scores_all
        scores_rf.append([i, train_acc_rf, validate_acc_rf])

        # turn to df
        scores_df2 = pd.DataFrame(scores_rf, columns=['max_depth_random_forest', 'train_acc_rf', 'validate_acc_rf'])

        # make new column
        scores_df2['difference'] = scores_df2.train_acc_rf - scores_df2.validate_acc_rf

        # sort on difference
        scores_df2.sort_values('difference')

        # print baseline
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(6,6))
    plt.plot(scores_df2.max_depth_random_forest, scores_df2.train_acc_rf, label='train', marker='o')
    plt.plot(scores_df2.max_depth_random_forest, scores_df2.validate_acc_rf, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.fill_between(scores_df2.max_depth_random_forest, scores_df2.train_acc_rf, scores_df2.validate_acc_rf, alpha=.4)
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,3, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(6,6))
    plt.bar(X_train.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()
    
    # KNN
    print(f"KNN")
    print(f"The number of features sent in : {len(X_train.columns)} and are {X_train.columns.tolist()}.")

    # run for loop and plot
    metrics = []
    for k in range(1, nn):
        
        # make the model
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the model
        knn.fit(X_train, y_train)
        
        # calculate accuracy
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']

    # make plottable df without the difference column
    metrics_plot = metrics_df.drop(columns='difference')
    print(f"{n} is the number of neighbors that produces the best fit model.")
    print(f"The accuracy score for the train model is {round(train_score,2)}.")
    print(f"The accuracy score for the validate model is {round(validate_score,2)}.")
            # print baseline
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}')
    
    
    # plot the data
    metrics_plot.set_index('k').plot(figsize = (10,10))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.axhline(y=train_score, color='blue', linestyle='--', linewidth=1, label='train accuracy')
    plt.axhline(y=validate_score, color='orange', linestyle='--', linewidth=1, label='validate accuracy')
    plt.fill_between(metrics_df.k, metrics_df['train score'], metrics_df['validate score'], alpha=.4)
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,nn,1))
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # LOGISTIC REGRESSION TRAIN
    print(f"LOGISTIC REGRESSION")
    print(f"Train Dataset")
    #create it
    logit = LogisticRegression(random_state=123)

    #fit it
    logit.fit(X_train, y_train)

    #use it
    lt_score = logit.score(X_train, y_train)
    print(f"The train model's accuracy is {round(lt_score,2)}")
    
    # print baseline
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}')
    
    #classification report
    print(classification_report(y_train, logit.predict(X_train)))

    # LOGISTIC REGRESSION VALIDATE
    print(f"LOGISTIC REGRESSION")
    print(f"Validate Dataset")
    #create it
    logit2 = LogisticRegression(random_state=123)

    #fit it
    logit2.fit(X_validate, y_validate)

    #use it
    lt_score2 = logit2.score(X_validate, y_validate)
    print(f"The validate model's accuracy is {round(lt_score2,2)}")

    # print baseline
    print(f'The baseline accuracy is {round((baseline_accuracy),2)*100}')
    
    #classification report
    print(classification_report(y_validate, logit2.predict(X_validate)))
    
#     performance = pd.concat([scores_df, scores_df2, metrics_df], axis=1)
    
#     return performance


# --- Regression

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def all_3_regression(X_train_r, y_train_r, X_validate_r, y_validate_r, X_test_r, y_test_r):
    """Contains all regression model code and returns df with results
    ---
    Format: metrics_df = functions()
    """
    print(f"Baseline Created")
    # establish baseline
    baseline = round(y_train_r.mean(),2)
    # make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(X_train_r))
    rmse, r2 = metrics_reg(y_train_r, baseline_array)

    # add to metrics df
    metrics_df = pd.DataFrame(data=[
    {'model':'-baseline-',
     'rmse':rmse.round(2),
     'r2':r2.round(2)}])

    print(f"OLS Created...")
    #intial ML model
    lr1 = LinearRegression()
    #make it
    rfe = RFE(lr1, n_features_to_select=1)
    #fit it
    rfe.fit(X_train_r, y_train_r)
    #use it on train
    X_train_rfe = rfe.transform(X_train_r)
    #use it on validate
    X_val_rfe = rfe.transform(X_validate_r)
    
    print(  'Top Linear Regression Feature:', rfe.get_feature_names_out())
    
    # build the model from the top feature
    #fit the thing
    lr1.fit(X_train_rfe, y_train_r)
    #use the thing (make predictions)
    pred_lr1 = lr1.predict(X_train_rfe)
    pred_val_lr1 = lr1.predict(X_val_rfe)
    # train
    metrics_reg(y_train_r, pred_lr1)
    # validate
    rmse, r2 = metrics_reg(y_validate_r, pred_val_lr1)

    # add to my metrics df
    metrics_df.loc[1] = ['ols', rmse.round(2), r2]

    # examine finding
    print(f'OLS: For every 1 dollar increase in Labor Sale, I predict a {lr1.coef_[0]:.2f} dollar increase in Profit')
  
    print(f"LassoLars Created...")
    #make it
    lars = LassoLars(alpha=1)

    #fit it
    lars.fit(X_train_r, y_train_r)

    #use it
    pred_lars = lars.predict(X_train_r)
    pred_val_lars = lars.predict(X_validate_r)
    pred_test_lars = lars.predict(X_test_r)
    
    #train
    metrics_reg(y_train_r, pred_lars)
    #validate
    rmse, r2 = metrics_reg(y_validate_r, pred_val_lars)
    
    #add to my metrics df
    metrics_df.loc[2] = ['lars', rmse.round(2), r2]
    
    print(f"GLM Created...")
    #make it
    glm = TweedieRegressor(power=0, alpha=1)

    #fit it
    glm.fit(X_train_r, y_train_r)

    #use it
    pred_glm = glm.predict(X_train_r)
    pred_val_glm = glm.predict(X_validate_r)
    #train
    metrics_reg(y_train_r, pred_glm)
    #validate
    rmse, r2 = metrics_reg(y_validate_r, pred_val_glm)
    
    # add to metrics df
    metrics_df.loc[3] = ['glm',rmse.round(2),r2]
    
    print(f"Best Model: LassoLars\nProceed to Test")
    # use the best model on test
    #use it
    rmse, r2 = metrics_reg(y_test_r, pred_test_lars)
    
    # add to metrics df
    metrics_df.loc[6] = ['test-lars',rmse.round(2),r2]
    
    # plot the predictions
    plot_model_predictions(pred_lr1, pred_lars, pred_glm, y_train_r, baseline)
    # plot the residuals
    plot_model_residuals(pred_lr1, pred_lars, pred_glm, y_train_r)
    # plot the actual and the predicted
    plot_model_actual_predicted(y_test_r,pred_test_lars)
    
    return metrics_df

# ---- VIZZES

def plot_target(df, col1):
    """This function plots the target variable"""
    # plot the target variable
    plt.figure(figsize=(10, 10))
    sns.histplot(df[col1], bins=60, color='purple')
    plt.axvline(df[col1].mean(), label='Average Profit', color='black', linewidth=2)
    plt.legend()
    plt.title('Distribution of Target Variable: Profit');

def plot_model_predictions(p1, p2, p3, y, baseline):
    """Plots the selected models predictions with baseline"""
    plt.scatter(p1, y, label='Linear Regression',  color="red", alpha=.6)
    plt.scatter(p2, y, label='LassoLars', color='green', alpha=.6)
    plt.scatter(p3, y, label='GLM', color='blue', alpha=.6)
    plt.plot(y, y, label='_nolegend_', color='grey')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81))

    plt.title("Where are predictions more extreme? More modest? Overfit?")
    plt.ylabel("Actual Profit")
    plt.xlabel("Predicted Profit")
    plt.legend()
    plt.show()
    
def plot_model_residuals(p1, p2, p3, y):
    """Plots the selected models residuals with baseline"""
    plt.axhline(label="No Error")

    plt.scatter(y, p1 - y, alpha=.5, color="red", label="Linear Regression")
    plt.scatter(y, p2 - y, alpha=.5, color="green", label="LassoLars")
    plt.scatter(y, p3 - y, alpha=.5, color="blue", label="GLM")

    plt.legend()
    plt.title("Do the size of errors change as the actual value changes?")
    plt.xlabel("Actual Profit")
    plt.ylabel("Residual: Predicted Profit - Actual Profit")
    plt.show()

def plot_model_actual_predicted(y, p1):
    """Plots the selected models predictions with actual"""
    plt.hist(y, color='purple', alpha=.4, label="Actual")
    plt.hist(p1, color='green', alpha=.9, label="LassoLars")
    # plt.hist(pred_glm, color='orange', alpha=.7, label='GLM')
    # plt.hist(pred_lars, color='yellow', alpha=.2, label='LassoLars')

    plt.xlabel("Profit Size")
    plt.ylabel("Count")
    plt.title("Comparing the Distribution of Actual to Predicted Profit Size")
    plt.legend()
    plt.show()