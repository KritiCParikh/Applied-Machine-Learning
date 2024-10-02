# Credit Default Prediction using XGBoost and Neural Network

**The importance of Credit Default prediction**

Solely in 2022, American Express amassed approximately $53 billion in revenue. TransUnion projected a surge in credit card delinquencies, foreseeing an increase from 2.1% at the end of 2022 to 2.6% by the conclusion of 2023. Anticipating which customers are most likely to default on their credit card accounts enables issuers to mitigate risk and exposure proactively.

Given the abundance of readily accessible customer data and many indicators, employing Machine Learning algorithms to forecast defaults presents a lucrative opportunity.

## Skills and Technologies:

* Python

* Data Analysis and Preprocessing

* Machine Learning

* Feature Selection

* Hyperparameter Tuning

* Model Evaluation

* Data Visualization

* Libraries: pandas, numpy, scikit-learn, xgboost, keras, tensorflow

### Business Impact:

- Credit Default Prediction: The model predicts the probability of credit card default, which can help financial institutions assess customer risk.
  
- Risk-Based Strategies: The code implements both conservative and aggressive strategies based on different prediction thresholds, allowing for flexible risk management.


**Data**

The historical data from credit card transactions encompasses 458,913 customers over a span of 13 months, covering 190 variables categorized into Payment, Spend, and Balance. Each month contains between 30,000 to 40,000 observations, and a percentage of customers defaulting in each month [23%, 28%]

Target Variable = 1 if the customer default on CC payment
                = 0 if the customer didn’t default

Dataset: https://www.kaggle.com/competitions/amex-default-prediction/overview

**Features**

All Features are divided into 5 categories: Delinquency, Payment, Balance, Risk & Spend 


**Feature Selection**

Built 2 XGBoost models to rank features according to their feature importance score. 

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/feat_imp-PhotoRoom.png-PhotoRoom.png)


**XGBoost - Grid Search**

The following combinations in the grid search:
1. Number of trees: 50, 100, and 300 :- 50 was to decrease the complexity and the variance, later we tried 300 for a lower bias 
2. Learning Rate: 0.01, 0.1 :-  0.1 = Conventional and 0.01 to validate if slower learning rate arrives at global minima smoothly without overshooting
3. % of observations used in each tree: 50%, 80% :- 50% for faster training & 80% to avoid overfitting
4. % of features used in each tree: 50%, 100% :- 50% again to avoid overfitting & faster training, and 100% for better results and low bias
5. Weight of default observations: 1, 5, 10 :– Since most of our non-default we need weights > 1

<img src = https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/Xgb%20plot1.png width = '500' height = '300'>

**Plot 1: Technically, Bias-Variance Tradeoff at X=0.94 & Y = 0.0075 (diff in Y is small, therefore lowest bias preferred)**

<img src = https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/XGB%20plot2.png width = '500' height = '300' style="float:right">

**Plot 2: Linear relationship between AUC train and test2, therefore highest AUC train preferred**


**Final XGBoost Model Parameters**

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/xgb%20final.png)


**Rank Ordering**

Here, in rank ordering, when we adjust the threshold upwards/increased, we observe that the default rate rises across higher threshold ranges.

<img src = https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/Rank_ordering.png width = '500' height = '300'>


**SHAP Analysis**

❒ BeeSwarm - Explains the cumulative impact of features on model 

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/beeswarm.png)

P_2 higher values drive the score down meaning higher the payment variable lower will be probability of default

Most features increase their impact on model output with higher feature value

❒ Waterfall - Explains prediction for specific observation  

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/waterfall.png)

Expected Model Output = -1.308, Output for 1100th customer = -4.311

P_2 singlehandedly drives prediction down by 1.26 whereas 37 other features collectively drive it down by 1.17


**Neural Network**

**NN Grid Search**

Combination of Hyper-Parameters in the grid search:
1. Number of hidden layers: 2, 4 :– 4 to increase the complexity & to get low bias and 2 for faster runtime 
2. Nodes in each hidden layer: 4, 6 :– 2 for simple neural network and 6 for complex neural network 
3. Activation function: ReLu, Tanh – ReLu isn’t saturated/zero-centered, tanh causes vanishing gradients 
4. Dropout regularization: 50%, 100% (no dropout) :– 50% to decrease complexity and avoid overfitting
5. Batch size: 100, 10000 :– 100 not low enough to overfit every batch and 10000 for faster processing 


<img src = https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/NN%20Plot1.png width = '500' height = '300'> <img src = https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/NN%20Plot%202.png width = '500' height = '300' style="float:right">


**Final Model**

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/final%20model.png)

Winners under various categories:

| Category      | Winner                                        |
| ------------- | --------------------------------------------- |
| Bias          | XGBoost                                       | 
| Variance      | Neural Network (diff in Std Dev is negligible)|
| Explanability | XGBoost (SHAP Analysis)                       |


**Strategy**

The conservative strategy has a lower threshold compared with aggressive one; hence accepts less applicants.

![image](https://github.com/KritiCParikh/Applied-Machine-Learning/blob/main/Graphs/ex%20strategy%20final.png)

0.5 :– Aggressive strategy because we want to increase our Revenue while maintaining the default rate below 10%

0.3 :– Conservative Strategy because the default decreases almost by half but revenue isn’t drastically affected
 
Thank You. Let’s keep learning and growing together!
