# 📈 Ecommerce-Inactive-Customers-Analysis-&-Prediction-Python

:warning: Customer churn leads to substantial profit loss in e-commerce businesses. 
This project aims to identify churn drivers and predict the probability of churn using a full machine learning pipeline—data preprocessing, EDA, feature engineering, modeling, and evaluation using Python libraries such as Pandas, Scikit-learn, and Matplotlib. 

<img width="979" height="578" alt="image" src="https://github.com/user-attachments/assets/55b63286-37b0-4275-a35f-f51d88f451b1" />
Author: Huy Huynh

Date: August 2025

Tool: Python 

---
## :bookmark: Table of contents:
---
## :memo: Background & Overview:

:dagger: Objective: 

This project investigates a customers characteristics dataset using Python in Google Colab, answering questions about churn drivers and behavior most related to the leaving of customers. 

- Finding top features strongly affects the inactivity of users
- Dive into these features, such as Tenure, Cashback, etc., to find some relationship
- Analyze key metrics engagement (login device, order category,...)
- Building a Machine Learning process and evaluating the best model to predict churn users
- Clustering churned users to uncover some properties for win-back strategies

:bow_and_arrow: Who is this project for:

- Data analysts & business intelligence professionals
- Marketing teams optimizing campaign performance
- Product managers and eCommerce decision-makers
---
## :page_facing_up: Dataset Description & Data Structure 
:clipboard: Data Source: 
- The dataset primarily used for this analysis is in the churn_prediction.xlsx file, containing detailed information about each customer's behavior
- Size: 1 Dataframe contains 20 features and more than 5500 rows
  
[Download the data here](https://docs.google.com/spreadsheets/d/1WDFxP3ipf-b2GnZJ1_8gtnrbBV08R2oH/edit?gid=216773957#gid=216773957)

:toolbox: Data Structure:
- Description of the table this project uses:

| Column Name                 | Description                                                                   | Data Type |
|-----------------------------|-------------------------------------------------------------------------------|-----------|
| CustomerID                  | Unique customer ID                                                            | int64     |
| Churn                       | Churn Flag                                                                    | int64     |
| Tenure                      | Tenure of customer in organization                                            | float64   |
| PreferredLoginDevice        | Preferred login device of customer                                            | object    |
| CityTier                    | City tier (1,2,3): 1 for big, 3 for small city                                | int64     |
| WarehouseToHome             | Distance between warehouse and home of customer                               | float64   |
| PreferPaymentMethod         | Preferred payment method of customer                                          | object    |
| Gender                      | Gender of customer                                                            | object    |
| HourSpendOnApp              | Number of hours spend on mobile application or website                        | float64   |
| NumberOfDeviceRegistered    | Total number of devices is registered on particular customer                  | int64     |
| PreferedOrderCat            | Preferred order category of customer in last month                            | object    |
| SatisfactionScore           | Satisfactory score of customer on service                                     | int64     |
| MaritalStatus               | Marital status of customer                                                    | object    |
| NumberOfAddress             | Total number of added address on particular customer                          | int64     |
| Complain                    | Any complaint has been raised in last month                                   | int64     |
| OrderAmountHikeFromlastYear | Percentage increases in order from last year                                  | float64   |
| CouponUsed                  | Total number of coupon has been used in last month                            | float64   |
| OrderCount                  | Total number of orders has been places in last month                          | int64     |
| DaySinceLastOrder           | Day since last order by customer                                              | float64   |
| CashbackAmount              | Average cashback in last month                                                | float64   |

## :open_book: Main process 
### 1. Data cleaning/preparation
1. Data loading and inspection
- Firstly, import some libraries like numpy, pandas, matplotlib, etc. for initial data exploration and preprocessing
- Secondly, import data and use functions to explore the dataset (.info, .shape, .nunique, .duplicated). Find out some information:
  - The dataset contains 5630 customers that can be observed
  - There is no duplicated row
  - Data type contains numeric and object data, which can be separately inspected later
2. Handling missing values
- Check for rows that have null value to see that 33% of them contain at least 1 null. So delete these rows is not an option
- Check for empty values each columns:

Columns                           |  Null_percentage  | Data_type          | unique_value
|---------------------------------|-------------------|--------------------|-------------
DaySinceLastOrder                 |  5.452931         | float64            | 22
OrderAmountHikeFromlastYear       |  4.706927         | float64            | 16
Tenure                            |  4.689165         | float64            | 36
OrderCount                        |  4.582593         | float64            | 16
CouponUsed                        |  4.547069         | float64            | 17
HourSpendOnApp                    |  4.529307         | float64            | 6
WarehouseToHome                   |  4.458259         | float64            | 34

- There are 7 columns that have null values, and the Null percentage of each one is approximately 5%. So the solution here is to replace them with indicated value
- Instead of choosing between mean, median, and mode to replace. Using an ML model and letting the machine determine the optimal value to fill in.
```python
# Using Random Forest model to predict null values
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

churn_filled = churn_raw.copy()
# Create loop through the churn_raw column
for col in churn_filled.columns:
  if churn_filled[col].isnull().sum() == 0:
    continue  # leave out the non null columns

  # Split the data into two parts: rows with values and rows with nulls
  know_df = churn_filled[churn_filled[col].notnull()]
  unknow_df = churn_filled[churn_filled[col].isnull()]

  # Create X_train, y_train Drop the target column (column has null) from features
  X_train = know_df.drop(columns = [col])
  y_train = know_df[col]
  X_pred = unknow_df.drop(columns = [col])

  # Fill other missing value with -999
  X_train = X_train.fillna(-999)
  X_pred = X_pred.fillna(-999)

  # Encode categorical columns with one-hot encoding
  X_train = pd.get_dummies(X_train)
  X_pred = pd.get_dummies(X_pred)

  # Align columns between train and prediction
  X_pred = X_pred.reindex(columns = X_train.columns, fill_value = 0)

  # Chose model to train:
  if y_train.dtype == 'object' or y_train.nunique() < 10:
    model = RandomForestClassifier(random_state = 42)
  else:
    model = RandomForestRegressor(random_state = 42)

  # Train and predict
  model.fit(X_train, y_train)
  predicted = model.predict(X_pred)

  # Fill the missing value with prediction
  churn_filled.loc[churn_raw[col].isnull(), col] = predicte
```

