# 📈 Ecommerce-Inactive-Customers-Analysis-&-Prediction-Python

:warning: Customer churn leads to substantial profit loss in e-commerce businesses. 
This project aims to identify churn drivers and predict the probability of churn using a full machine learning pipeline using Python. 

<img width="979" height="578" alt="image" src="https://github.com/user-attachments/assets/55b63286-37b0-4275-a35f-f51d88f451b1" />
- Author: Huy Huynh
- Date: August 2025
- Tool: Python 

---
## :bookmark: Table of contents:
---
## :memo: Background & Overview:

:dagger: Objective: 

This project investigates a customer characteristics dataset using Python in Google Colab, answering questions about churn drivers and behavior most related to the leaving of customers. 

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
**1. Data loading and inspection**
- Firstly, import some libraries like numpy, pandas, matplotlib, etc. for initial data exploration and preprocessing
- Secondly, import data and use functions to explore the dataset (.info, .shape, .nunique, .duplicated). Find out some information:
  - The dataset contains 5630 customers that can be observed
  - There is no duplicated row
  - Data type contains numeric and object data, which can be separately inspected later
  
**2. Handling missing values**
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

- 7 columns have null values, and the Null percentage of each one is approximately 5%. So the solution here is to replace them with indicated value
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
**3. Data distribution & Outliers**
**Numeric columns:**
- Investigate outliers in numeric columns using boxplot & histogram.
- 4 columns have unreasonably large outliers that could be removed (the outliers)

**Tenure:**
  <img width="1098" height="653" alt="image" src="https://github.com/user-attachments/assets/8590bc29-66b3-4985-baf2-d7f0ff371394" />
**WarehouseToHome:**
<img width="1106" height="654" alt="image" src="https://github.com/user-attachments/assets/d7ae5abd-1c3d-4f4b-83b8-e79678c8468a" />
**NumberOfAddress:**
<img width="1107" height="655" alt="image" src="https://github.com/user-attachments/assets/33be1d41-66ad-453a-9d34-c94ecf765b20" />
**DaySinceLastOrder:**
<img width="1121" height="664" alt="image" src="https://github.com/user-attachments/assets/fcbe5c64-ad15-4325-aec2-fd2a10fb1ad5" />

- Using the IQR algorithm to filter outliers:
```python
# Remove outliers of these 4 numeric columns
cols = ['Tenure','WarehouseToHome','DaySinceLastOrder','NumberOfAddress']

churn_filtered = churn_filled.copy()

# Using IQR to indicate outliers
for col in cols:
    Q1 = churn_filtered[col].quantile(0.25)
    Q3 = churn_filtered[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    churn_filtered = churn_filtered[(churn_filtered[col] >= lower) & (churn_filtered[col] <= upper)]

# Draw a box plot to check the remove of outliers

plt.figure(figsize=(8, 6))
sns.boxplot(data =  churn_filtered[cols], color = 'skyblue')
plt.title('Box Plot of 4 Features')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Shape of df after remove outliers
print(f"After removing outliers: {churn_filtered.shape}")
```
<img width="784" height="604" alt="image" src="https://github.com/user-attachments/assets/405314d3-5d9f-44c3-930a-a23c1be190ab" />

**Category columns:** 
- Investigate category columns:

<pre>```
unique value of PreferredLoginDevice:
['Mobile Phone' 'Computer' 'Phone']
unique value of PreferredPaymentMode:
['E wallet' 'Cash on Delivery' 'Debit Card' 'Credit Card' 'COD' 'CC' 'UPI']
unique value of Gender:
['Female' 'Male']
unique value of PreferedOrderCat:
['Fashion' 'Laptop & Accessory' 'Mobile Phone' 'Grocery' 'Mobile' 'Others']
unique value of MaritalStatus:
['Married' 'Single' 'Divorced'] 
```</pre>

- We can see that the data has some mistakes over similar features. For example, there are 2 types of payment methods which have the same meaning: "cash on Delivery" and "COD". So here is the fix:
  - PreferredLoginDevice: Phone = Mobile Phone  
  - PreferredPaymentMode: Credit Card = CC, and Cash On Delivery = COD
  - PreferedOrderCat: Mobile = Mobile Phone

### 2. Exploratory Data Analysis
**a. Top important features:**
```python
# ML model to measure features importances
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Prepare data
X = churn_df.drop('Churn', axis = 1)
y = churn_df['Churn']
X_encoded = pd.get_dummies(X, drop_first = True)
# print(X.shape,X_encoded.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.3, stratify = y, random_state = 42 )

# Create train model
model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

# Get Feature Importance
importances = model.feature_importances_
feature_names = X_train.columns

# Create and sort dataframe for drawing barplot
importance_df = pd.DataFrame({
    'Feature' : feature_names,
    'Importances' : importances
}).sort_values(by = 'Importances', ascending = False)

# Draw a plot for the importance data with the original categories (without encode)
import re

# Function to extract base (original) feature name
def get_base_feature(col):
    return re.split(r'_[^_]+$', col)[0] if '_' in col else col

# Add base feature column
importance_df['BaseFeature'] = importance_df['Feature'].apply(get_base_feature)
# print(importance_df)

# Group importances by base feature and sum the the importances we have split 
grouped_importance = importance_df.groupby('BaseFeature')['Importances'].sum().reset_index()
grouped_importance = grouped_importance.sort_values(by='Importances', ascending=False)

# Plot grouped feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importances', y='BaseFeature', data=grouped_importance)
plt.title('Importance of Features to customer Churn')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
```
<img width="1280" height="735" alt="image" src="https://github.com/user-attachments/assets/870840ab-532d-403c-ac01-c0382ccef39c" />

- So the top 5 things to effect the most on the churn is:
  - Tenure,
  - Chash back,
  - Distance from warehouse to customer,
  - Complain
  - Day since last order.
- Dive into these features and find some insight:

**Tenure:**
<img width="1001" height="604" alt="image" src="https://github.com/user-attachments/assets/a7434643-4c78-4a47-b13c-9b2e56afea92" />

- For the Tenure: These users who use the app for 1 month or less have a churn rate of 53%, but the users who last more than 6 months have very little intention to quit, just less than 6%.
  - So the recommendation is to make customers want to extend their time using the company's services
  
**CashbackAmount:**
<img width="1012" height="600" alt="image" src="https://github.com/user-attachments/assets/12b9f739-a3c5-48e0-8a7d-01b405dce9af" />

- For the CashBack Amount: The churn rate of those who got the cashback >163$ is much lower than those who got less cashback
  - But for more information, get features about discounts, gifts, and vouchers to understand more about the trend
  
**WarehouseToHome:**
<img width="998" height="591" alt="image" src="https://github.com/user-attachments/assets/35213a83-87eb-4f16-a02e-35ef059023b6" />

- For the Warehouse to Home: The farther the distance from warehouses to customers' homes, the more likely they will leave.
  - Investigate the delivery duration to understand the reason for that trend, concentrate on those people who have a home more than 14km away in terms of distance.
  
**DaySinceLastOrder:**
<img width="1003" height="602" alt="image" src="https://github.com/user-attachments/assets/c03f1319-009b-4b9c-9504-a27fc0d1b66c" />

- For DaysinceLastOrder: A lot of People who make orders in just less than 2 days churned, but those people who didn't make any purchase recently are still stuck.
  - Find out what is happening with the UX/UI, any malfunctions or disturbances during purchase.
  - For those people who did not quit but haven't bought anything for several days -> perhaps run a special welcome campaign
    
**Complain:**

<img width="899" height="611" alt="image" src="https://github.com/user-attachments/assets/99803e73-7100-4445-a313-fdf8e1103e16" />

- For Complain: People who complain tend to churn more
  - Did the complaint dealt with correctly?

**b. Categorical features:**
- Plot the churn percentage for each category to see the trend:
  
**PreferedLoginDevice:**
  <img width="888" height="566" alt="image" src="https://github.com/user-attachments/assets/32ba662f-be53-457c-9a7c-b862f647954d" />

- Login Device: People using computers tend to churn slightly higher than mobile users.
  - Are there some bugs in the computer version of the app/web?
  - PC User interface might not be as good as on the phone?

**PreferedPaymentMethod:**
  <img width="889" height="607" alt="image" src="https://github.com/user-attachments/assets/27588c75-12f5-4b89-8fa7-a663c71e4ffe" />

- Payment Method: the COD and E-Wallet payment mode churn percent was very high, above 23%
  - User who prefer COD might not fully trust the service -> Can change their behavior to prepayment by some voucher
  - The E-Wallet user could purchase because of the coupons in the e-wallet -> No coupons, no purchase 
  
**Gender:**

<img width="882" height="599" alt="image" src="https://github.com/user-attachments/assets/3d5e9ff1-7bbf-4c7e-84e4-f500d0a7fe6d" />

- Gender: The Male churn is just a little bit higher than female.
  - We should investigate whether they are gay or not

**PreferedOrderCat:** 

<img width="924" height="599" alt="image" src="https://github.com/user-attachments/assets/d5539604-94b7-4873-9d6d-69694fc17685" />

- Prefer order category: Those who prefer purchasing mobile phones tend to churn more frequently, at about 27.5%. The Fashion customers' churn rate is kinda high too, about 15.5%
  - Phone is a high-value product, so those customers are just one-time buyers. So we can give them some vouchers if they bring some of their relations to join the company's services.
  - As for the Fashion category, learn more about the quality of the products. Many customers stop buying clothes from a shop because of its poor products.
  
**Marital Status:**

<img width="864" height="595" alt="image" src="https://github.com/user-attachments/assets/bbaad357-ef12-4263-8e1f-d856cccb716b" />

- Marital status: Single people have a churn percentage much higher than those with different marital statuses.
  - Why doesn't our service attach those unmarried people?

### 3. Building Machine Learning model to predict churn users.







