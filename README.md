# Personal_Project_Shop_Talk

# Project Description

This repo contains a personal project looking at a mechanic shops revenue and costs.

# Project Goal

* Find the key drivers of profit for friend's mechanic shop

# Initial Thoughts

My initial thoughts were correct! Parts and labor sales would affect profit the greatest.

# The Plan

* Acquire
    * Data acquired from Jeff's local machine, emailed to me
    * Each row represents a customer's visit
    * Each column represents a feature of the shop's business
  
* Prepare
    * 10880 rows × 12 columns before cleaning
    * 6412 rows × 10 columns after cleaning
        * 59% Original Remaining
    * Dropped negative index rows
    * Changed date to datetime type
    * Changed numeric columns to floats, round to 2
    * Feature engineereed columns: Profit Per Part, Profit Per Labor, Profit
    * Binned to create order sizes: Small, Medium, Large [0, 150, 750, 2000]
    * Dropped columns:
        * sublet_cost, sublet_sale, sale_total, total_cost, percent_profit, amount_profit
    * nulls and NaN's - dropped from labor_cost and feature engineered columns
    * Outliers removed: Anything greater than 2000 were dropped
            * Point was to "zoom" in on average, everyday business to gather actionable insights
    * Encode for Categorical
    * Split into Train, Validate, and Test
            
* Explore
    * Questions
        * Are the means of labor cost and profit the same?
        * Are the means of labor cost and profit the same controlling for profit size?
            * Diving deeper from the previous question
        * Are the means of parts cost and profit the same?
        * Are the means between parts cost and profit the same controlling for profit size?
            * Diving deeper from the previous question
    
* Model
    * **Features to send in:**<br>
    Classification:
    * labor_cost
    * labor_sale
    * parts_sale
    * parts_cost

    Regression:
    * labor_cost
    * labor_sale
    * parts_sale
    * parts_cost

    ***Dropped profit_per_part and profit_per_labor features used for exploration***

    **Models Selected:**<br>

    Classification:
    * KNN
    * Decision Tree
    * Random Forest
    * Logisitic Regression

    Regression:
    * OLS (Linear Regression)
    * LassoLars
    * GLM (Tweedie)

    * Conclusions

# Data Dictionary  

| Feature | Definition|
|:--------|:-----------|
|date| Date of Transaction|   
|customer| Customer Name|
|parts_cost| Cost to Shop to Order Part|       
|labor_cost| Cost to Shop to Payout Labor|              
|total_cost| Parts + Labor Cost to Shop|       
|parts_sale| Profit to Shop Selling Part|
|labor_sale| Profit to Shop Charging Labor|   
|sale_total| Parts + Labor Profit to Shop|
|profit_per_part| Profit Per Part Per Job|
|profit_per_labor| Profit Per Labor Per Job|
|profit| Sale Total - Total Cost|

# Steps to Reproduce
1. Clone this repo
2. Run notebook

---

# Takeaways

Explore takeaways pointed at some interesting variations presenting when the profits got higher:

1. Null was rejected. The means of labor cost and profit are not the same. I notice the confidence interval flares at the tip indicating there is a variance occuring there. The average cost for labor and the average profit leaves room for improvement. On the smaller scale, it seems to be relatively healthy. The higher the profit goes, the mean of labor cost stays relatively low indicating a healthy relationship between cost for the shop to employ techs and the profit to be made.
> * 81% correaltion<br>

2. Null was rejected. I dove deeper into the first question here, controlling for profit size. Overall, the smaller jobs trend positive, the medium jobs do not trend as positive. I would like to see as costs go up the profit would go up more than it does. It kind of clumps around 100 dollars for labor cost and 250 dollars for profits which matches the business knowledge given to me by the owner of a labor charge of 108/hr and the mean profit here being 244. The larger jobs have almost flatlined with no confidence for prediction at all. There is lots of variance here. The focus here is on labor cost to profit.
> * 81% correlation<br>

3. Null was rejected. Based on the graph I can see that there is some amount of variation causing some interesting spread. The confidence intervals vary widely indicating a greater amount of distance between points as they get larger. This is interesting, the next question I dove a little deeper into the possible 'why' behind this happening.
> * 85% correlation<br>

4. Null was rejected. Diving deeper into question three and controlling for profit size. The smaller jobs tend to be trending excellent, the medium size jobs have increased variation with a decrease in positive trendline, the large jobs have almost completely flatlined. This was an interesting find. The focus here is on parts cost to profit.
> * 85% correlation<br>

Modeling:

* **Classification**
    * Accuracy is the metric
    * Target is profit_size
    * ***Baseline: 62%***
    * Decision Tree, depth 1
        * 91% Train
        * 95% Validate
    * Decision Tree, depth 2
        * 91% Train
        * 95* Validate
    * Random Forest, depth 1
        * 91% Train
        * 95% Validate
    * Random Forest, depth 2
        * 91% Train
        * 95% Validate
    * KNN, 1
        * 96% Train
        * 97% Validate
    * Logistic Regression
        * 98% Train
        * 98% Validate
        
Classification overfit on every model except for KNN possibly providing a decent outcome. If we go this route, would need to reintroduce more original data to possibly get a better fit with more than 3 bins.
    
* **Regression**
    * RMSE and R2 are the metrics
    * Target is profit
    * ***Baseline RMSE: $280.59***
    * ***Baseline R2:   0***

    * OLS Standard Linear Regression:
        * $72.53
        * .91   
    * LARS:
        * $65.29
        * .93
    * GLM:
        * $12.74
        * 100
        
***LassoLars Model selected to test over all models created - performed great on test! RMSE of 70.42 and an R2 of .93***
    
---

***Overall Takeaways:***
From all three iterations, this being the final, I noticed a trend of undercharging for parts and labor. From iterations one and two, I discovered that average part sales are 9.54 too low (iteration one) and that labor charges are 15.23 too low (iteration two).
    * I got to these numbers by taking the averages between iterations given different scenarios. The industry markup average is 100% for parts sales and other shops in the valley charge closer to 130 per labor hour.
    
- Of note: parts cost after ~$1,100 saw a significant drop in parts sale, almost like someone is a good guy and giving out discounts... :)

# Recommendation
    
* For the data scientists: Add outliers back in to beef up the data set and see how it performs. I expect the model to perform well on future data and look forward to the next steps to provide that for the customer.
    
* For the business: Recommend waiting for next steps to be completed before proceeding. The business is well in the green, it's safe to wait for additional analysis to be finished before raising costs.
    
# Next Steps:
    
* Perform time series analysis to provide predictions for profit gains if the shop were to act on the insights provided above. Make Jeff more money!
