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

| Column Name                 | Description                                                                  | Data Type |
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

---
## :open_book: Main process 
### 1. Data preparation


