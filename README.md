# Predict-House-Price

## Description
The advancement in big data industry has increased the importance of machine learning applications for enhancing decision-making process due to its predictive power. One of the common applications is predicting house price. It is useful for housing developers to set suitable house prices based on their qualities as well as customers to determine their budgets and timings for purchases. 

Therefore, I covered some statistical studies, linear regression analyses, and non-linear machine learning algorithms (Random Forest, Lasso, Ridge, and XGBoost) for predicting house prices and investigating the key features deciding house prices. I selected a Kaggle dataset about house prices in Ames and Iowa. Please click the deployed model result on the link below:
[Deployment Link](https://huggingface.co/spaces/catherinehelenna/House-Price-Predictor)

## Data Source
a. **House Prices**: Main dataset from 2016.
   - The dataset ”House Prices - Advanced Regression Techniques” consists of 1,459 entries with 81 variables representing house prices for properties built between 1872 and 2010. These variables, also referred to as features, are used to predict the sale price (”SalePrice”) of a property.
   - [Link to Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Analysis and Modeling Details
The data analysis was divided into three parts:
1. Distribution analysis: sale price distribution, normality test between log-transformed and original sale price, confirmatory test (statistical significance test between sale price and categorical features).
2. Correlation analysis: Pearson and Spearman correlation tests.
3. Regression analysis: OLS results comparison between log-transformed sale price vs. original sale price.
4. Modeling algorithms: compare the MSE results of selected linear and non-linear models for house sale price prediction.

## Results
1. Highlights on distribution analysis
- Sale price distribution on the dataset indicated right-skewness, with median concentrated around
$162,000. 
- After applying a log transformation to the sale prices, the distribution became notably more symmetrical.
-  The Kruskal-Wallis test could filter the categorical variables which had no significant relationship
with sale price: Street, PoolQC, LandSlope, Utilities.

2. Highlights on correlation analysis
- Some features were redundant so they were combined into a single feature.
```python
df['TotalHouseSF'] = df['GrLivArea'] + df['TotalBsmtSF'] +df['1stFlrSF']
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
```
3. Highlights on regression analysis (OLS)
- The log transformed sale price had slightly better R-2 performance than original one (0.945 > 0.935).
- The F-statistics of log transformed sale price was 66.64 with p-value < 0.01 implying at least one of the predictors significantly associated with the log transformed sale price.

4. Higlights on modeling algorithms:
- In validation set, the Lasso Regression achieved the best performance due to highest R-2 score (0.892494) and lowest MSE of log-transformed price (0.003300).
- In test set, the Lasso Regression consistently performed like in validation with R-2 score of 0.9316 and MSE of sale price at 306898281.3212.
