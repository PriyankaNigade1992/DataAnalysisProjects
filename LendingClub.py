#!/usr/bin/env python
# coding: utf-8

# ### Case Study #1
# 
# priyankamohanrao.nigade@pace.edu

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost


# In[2]:


# Loading the data into Data Frame

loan_df = pd.read_csv('loans_full_schema.csv')

print(loan_df.shape)
loan_df.head()


# There are total 10000 instances(rows) and  55 attributes (columns) in the dataset.

# #### Describe the dataset
# 
# <li> There are 55 columns and 10000 rows in the given dataset
# <li> Most of the column values are numeric
# <li> state, homeownership, application_type, verified_income, loan_status, initial_listing_status, disbursement_method,grade,loan_purpose, issue_month are categorical columns.
# 
# *** Column info***
# <li>emp_title - 
# Job title.
# 
# <li>emp_length - 
# Number of years in the job, rounded down. If longer than 10 years, then this is represented by the value 10.
# 
# <li>state - 
# Two-letter state code.
# 
# <li>home_ownership - 
# The ownership status of the applicant's residence.
# 
# <li>annual_income - 
# Annual income.
# 
# <li>verified_income - 
# Type of verification of the applicant's income.
# 
# <li>debt_to_income - 
# Debt-to-income ratio.
# 
# <li>annual_income_joint - 
# If this is a joint application, then the annual income of the two parties applying.
# 
# <li>verification_income_joint - 
# Type of verification of the joint income.
# 
# <li>debt_to_income_joint - 
# Debt-to-income ratio for the two parties.
# 
# <li>delinq_2y - 
# Delinquencies on lines of credit in the last 2 years.
# 
# <li>months_since_last_delinq - 
# Months since the last delinquency.
# 
# <li>earliest_credit_line - 
# Year of the applicant's earliest line of credit
# 
# <li>inquiries_last_12m - 
# Inquiries into the applicant's credit during the last 12 months.
# 
# <li>total_credit_lines - 
# Total number of credit lines in this applicant's credit history.
# 
# <li>open_credit_lines - 
# Number of currently open lines of credit.
# 
# <li>total_credit_limit - 
# Total available credit, e.g. if only credit cards, then the total of all the credit limits. This excludes a mortgage.
# 
# <li>total_credit_utilized - 
# Total credit balance, excluding a mortgage.
# 
# <li>num_collections_last_12m - 
# Number of collections in the last 12 months. This excludes medical collections.
# 
# <li>num_historical_failed_to_pay - 
# The number of derogatory public records, which roughly means the number of times the applicant failed to pay.
# 
# <li>months_since_90d_late - 
# Months since the last time the applicant was 90 days late on a payment.
# 
# <li>current_accounts_delinq - 
# Number of accounts where the applicant is currently delinquent.
# 
# <li>total_collection_amount_ever - 
# The total amount that the applicant has had against them in collections.
# 
# <li>current_installment_accounts - 
# Number of installment accounts, which are (roughly) accounts with a fixed payment amount and period. A typical example might be a 36-month car loan.
# 
# <li>accounts_opened_24m - 
# Number of new lines of credit opened in the last 24 months.
# 
# <li>months_since_last_credit_inquiry- 
# Number of months since the last credit inquiry on this applicant.
# 
# <li>num_satisfactory_accounts - 
# Number of satisfactory accounts.
# 
# <li>num_accounts_120d_past_due - 
# Number of current accounts that are 120 days past due.
# 
# <li>num_accounts_30d_past_due - 
# Number of current accounts that are 30 days past due.
# 
# <li>num_active_debit_accounts - 
# Number of currently active bank cards.
# 
# <li>total_debit_limit - 
# Total of all bank card limits.
# 
# <li>num_total_cc_accounts - 
# Total number of credit card accounts in the applicant's history.
# 
# <li>num_open_cc_accounts - 
# Total number of currently open credit card accounts.
# 
# <li>num_cc_carrying_balance - 
# Number of credit cards that are carrying a balance.
# 
# <li>num_mort_accounts - 
# Number of mortgage accounts.
# 
# <li>account_never_delinq_percent - 
# Percent of all lines of credit where the applicant was never delinquent.
# 
# <li>tax_liens - 
# a numeric vector
# 
# <li>public_record_bankrupt - 
# Number of bankruptcies listed in the public record for this applicant.
# 
# <li>loan_purpose - 
# The category for the purpose of the loan.
# 
# <li>application_type - 
# The type of application: either individual or joint.
# 
# <li>loan_amount - 
# The amount of the loan the applicant received.
# 
# <li>term - 
# The number of months of the loan the applicant received.
# 
# <li>interest_rate - 
# Interest rate of the loan the applicant received.
# 
# <li>installment - 
# Monthly payment for the loan the applicant received.
# 
# <li>grade - 
# Grade associated with the loan.
# 
# <li>sub_grade - 
# Detailed grade associated with the loan.
# 
# <li>issue_month - 
# Month the loan was issued.
# 
# <li>loan_status - 
# Status of the loan.
# 
# <li>initial_listing_status -
# Initial listing status of the loan. (I think this has to do with whether the lender provided the entire loan or if the loan is across multiple lenders.)
# 
# <li>disbursement_method -
# Dispersement method of the loan.
# 
# <li>balance - 
# Current balance on the loan.
# 
# <li>paid_total - 
# Total that has been paid on the loan by the applicant.
# 
# <li>paid_principal - 
# The difference between the original loan amount and the current balance on the loan.
# 
# <li>paid_interest - 
# The amount of interest paid so far by the applicant.
# 
# <li>paid_late_fees - 
# Late fees paid by the applicant.

# #### Any issues with Dataset
# 
# <li> Too many missing values

# ### Explorartory Data Analysis

# In[3]:


# Given - data set only represents loans actually made
# Lets plot counter plot to check the loan status count 

plt.figure(figsize=(20, 8))
sns.countplot(loan_df.loan_status)


# From above plot we can see many loans are in current state

# In[4]:



plt.figure(figsize=(20, 8))
sns.distplot(loan_df['interest_rate'],bins=40)


# In[5]:


plt.figure(figsize=(18,7))
sns.countplot(x="loan_purpose", data=loan_df)
plt.show() 


# Many loans are taken for the purpose of debt consolidation

# In[6]:


plt.figure(figsize=(20, 8))

# Plot
plt.scatter(loan_df['loan_amount'], loan_df['annual_income'], alpha=0.5)
plt.title('Scatter plot loan amount vs annual income')
plt.xlabel('loam amount')
plt.ylabel('annual income')
plt.show()


# In[7]:


# check outliers using bocplot
plt.figure(figsize=(17, 7))
sns.boxplot(x='loan_status', y='loan_amount', data=loan_df)
sns.despine()


# In[8]:


f, axes = plt.subplots(1, 2, figsize=(15,5), gridspec_kw={'width_ratios': [1, 2]})
sns.countplot(x='grade', hue='loan_status', data=loan_df, order=sorted(loan_df['grade'].unique()), palette='seismic', ax=axes[0])
sns.countplot(x='sub_grade', data=loan_df, palette='seismic', order=sorted(loan_df['sub_grade'].unique()), ax=axes[1])
sns.despine()
axes[0].set(xlabel='Grade', ylabel='Count')
axes[0].set_title('Count of Loan Status per Grade', size=20)
axes[1].set(xlabel='Sub Grade', ylabel='Count')
axes[1].set_title('Count of Loan Status per Sub Grade', size=20)
plt.tight_layout()


# In above plot the sub grades in blue are the good ones and in red are either late or charged off.

# In[9]:


#correlation with respect to label 'interest_rate' and filtering out labels with less correlation as we have many columns

correlation = loan_df.corr(method='pearson', min_periods=1)
columns = correlation[abs(correlation['interest_rate']) > 0.1]['interest_rate']

column_list = columns.index

high_corr_df = pd.DataFrame(loan_df, columns = column_list)

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(high_corr_df.corr(), annot=True, cmap=plt.cm.Greens)
plt.show()


# In[10]:


# Check the column names, datatypes and the non-null value counts.

loan_df.info()


# ### Handle Missing Values

# In[11]:


# Check count of null values in each column

loan_df.isna().sum()


# #### Columns with many null values
# 
# <li> annual_income_joint            =     8505 
# <li> verification_income_joint      =     8545
# <li> debt_to_income_joint           =     8505
# <li> months_since_last_delinq       =     5658
# <li> months_since_90d_late          =     7715

# In[12]:


# Create a new Dataframe data processing so that we won't make any changes in original Dataframe

loan_lend_club_df = loan_df.copy()


# In[13]:


# Handling NA values with fillna(0) for columns - annual_income_joint, debt_to_income_joint, 
# months_since_last_delinq,months_since_90d_late

# loan_lend_club_df['annual_income_joint'] = loan_lend_club_df['annual_income_joint'].fillna(0)
# loan_lend_club_df['debt_to_income_joint'] = loan_lend_club_df['debt_to_income_joint'].fillna(0)
# loan_lend_club_df['months_since_last_delinq'] = loan_lend_club_df['months_since_last_delinq'].fillna(0)
# loan_lend_club_df['months_since_90d_late'] = loan_lend_club_df['months_since_90d_late'].fillna(0)


# From above datset information we can see below column contains more than 50% (total records =10000) null values 
# <li>verification_income_joint
# <li>annual_income_joint
# <li>debt_to_income_joint
# <li>months_since_90d_late
# <li>months_since_last_delinq
#     
# We cannot replace these null values, so lets delete these columns.    

# In[14]:


loan_lend_club_df.drop(['verification_income_joint','annual_income_joint','debt_to_income_joint','months_since_last_delinq','months_since_90d_late','months_since_last_delinq'],
                      axis=1, inplace = True)


# In[15]:


# Handle Null values of emp_title column

emp_titles = list(loan_lend_club_df['emp_title'].unique())

len(emp_titles) # or loan_lend_club_df['emp_title'].nunique()


# In[16]:


# As there are too many categorical values which cannot be handled by dummy variables and 
# thus we cant use this column to build the model so lets drop it.

loan_lend_club_df.drop('emp_title', axis = 1, inplace = True)


# In[17]:


# From correlation matrix we can see emp_length column has no impact on target (interest_rate) column, so we can drop it

loan_lend_club_df.drop('emp_length', axis = 1, inplace = True)


# In[18]:


loan_lend_club_df.isna().sum()


# Raplace null values of months_since_last_credit_inquiry and num_accounts_120d_past_due columns with mean value

# In[19]:



loan_lend_club_df['months_since_last_credit_inquiry'].fillna(int(loan_lend_club_df['months_since_last_credit_inquiry'].mean()), inplace=True)


# In[20]:



loan_lend_club_df['num_accounts_120d_past_due'].fillna(int(loan_lend_club_df['num_accounts_120d_past_due'].mean()), inplace=True)


# In[21]:


# debt_to_income contains small number of null values lets drop null values
loan_lend_club_df.dropna()


# In[22]:


loan_lend_club_df.isna().sum()


# ### Handling categorical values
# 
# <li> state	
# <li> homeownership	
# <li> application_type
# <li> verified_income
# <li> loan_status
# <li> initial_listing_status	
# <li> disbursement_method
# <li> grade
# <li> loan_purpose
# <li> issue_month

# In[23]:


# state and grade columns have high number of categorical values, so we can drop them

loan_lend_club_df.drop('state', axis = 1, inplace= True)
loan_lend_club_df.drop('sub_grade', axis = 1, inplace= True)


# In[24]:


loan_lend_club_df['homeownership'].value_counts()


# In[25]:


# Replacing the categorical values 

loan_lend_club_df.replace({'homeownership': {"MORTGAGE": 0,'RENT':1, 'OWN':2}},inplace=True)


# In[26]:


loan_lend_club_df['application_type'].value_counts()


# In[27]:


# Replacing the categorical values 

loan_lend_club_df.replace({'application_type': {"individual": 0,'joint':1}},inplace=True)


# In[28]:


loan_lend_club_df['verified_income'].value_counts()


# In[29]:


# Replacing the categorical values 

loan_lend_club_df.replace({'verified_income': {"Not Verified": 0,'Verified':1, 'Source Verified':2}},inplace=True)


# In[30]:


loan_lend_club_df['loan_status'].value_counts()


# In[31]:


# Replacing the categorical values 

loan_lend_club_df.replace({'loan_status': {"Current": 0,'Fully Paid':1, 'In Grace Period':2, 'Late (31-120 days)':3, 
                                          'Late (16-30 days)':4, 'Charged Off': 5}},inplace=True)


# In[32]:


loan_lend_club_df['initial_listing_status'].value_counts()


# In[33]:


# Replacing the categorical values 

loan_lend_club_df.replace({'initial_listing_status': {"whole": 0,'fractional':1}},inplace=True)


# In[34]:


loan_lend_club_df['disbursement_method'].value_counts()


# In[35]:


# Replacing the categorical values 

loan_lend_club_df.replace({'disbursement_method': {"Cash": 0,'DirectPay':1}},inplace=True)


# In[36]:


loan_lend_club_df['grade'].value_counts()


# In[37]:


# Replacing the categorical values 

loan_lend_club_df.replace({'grade': {"A": 0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}},inplace=True)


# In[38]:


loan_lend_club_df['loan_purpose'].value_counts()


# In[39]:


# Replacing the categorical values 

loan_lend_club_df.replace({'loan_purpose': {"debt_consolidation": 0,'credit_card':1,'other':2,'home_improvement':3,
                                            'major_purchase':4,'medical':5,'house':6,'car':7,'small_business':8,'moving':9,
                                           'vacation':10,'renewable_energy':11}},inplace=True)


# In[40]:


loan_lend_club_df.drop('issue_month', axis=1, inplace = True)


# In[41]:


# The debt to income has very less null values.
loan_lend_club_df = loan_lend_club_df.dropna() 


# In[42]:


# Final Dataset

loan_lend_club_df


# In[43]:


#correlation with respect to label 'interest_rate' and filtering out labels with less correlation as we have many columns

correlation = loan_lend_club_df.corr(method='pearson', min_periods=1)
columns = correlation[abs(correlation['interest_rate']) > 0.1]['interest_rate']

column_list = columns.index

high_corr_df = pd.DataFrame(loan_lend_club_df, columns = column_list)

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(high_corr_df.corr(), annot=True, cmap=plt.cm.Oranges)
plt.show()


# In[44]:


corr = loan_lend_club_df.corr()['interest_rate'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', corr.tail(10))
print('\nMost Negative Correlations:\n', corr.head(10))


# ## Applying  Model

# #### Divide Dataset into train and test dataset (70%:30%) ration

# In[45]:


label = loan_lend_club_df['interest_rate'] # interest_rate is our target column
features_col = loan_lend_club_df.drop(['interest_rate'], axis = 1) 

# Dividing the dataset in train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(features_col, label, test_size = 0.3, random_state = 40)


# In[46]:


# Function to evaluate model performance
def evaluate_model(y_test, y_pred):

    r_squared = r2_score(y_test, y_pred)
    print('R Squared:', round(r_squared, 2))
    
    mse =mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', round(mse, 2))
 
    rmse = np.sqrt(mse)
    print('Root Mean Squared Error:', round(rmse,2))
    
    mape = calculate_mape(y_test, y_pred)
    #print the MAPE value
    print('The Mean Absolute Percentage Error (MAPE) value: ', round(mape,2))
    
     # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print(f'Accuracy: {round(accuracy, 2)} %')


# In[47]:


# Function to calculate the Mean Absolute Percentage Error (MAPE) value
def calculate_mape(actual,pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100


# ### Linear Regression

# In[48]:


# Building Linear Regressor  model 
linear_regressor = LinearRegression()

# fit the model with data
linear_regressor.fit(X_train, y_train)

# Predict the Xtest data with Linear Regressor model 
y_pred = linear_regressor.predict(X_test)


# In[49]:


score=linear_regressor.score(X_train, y_train)
print('Model score:', score)
 


# In[50]:


evaluate_model(y_test, y_pred)


# ### Random Forest Regressor

# In[51]:


# Building Random Forest Regressor  model 
rf_regressor = RandomForestRegressor()

# fit the model with data
rf_regressor.fit(X_train, y_train)

# Predict the Xtest data with Random Forest Regressor model 
y_pred = rf_regressor.predict(X_test)


# In[52]:


score= rf_regressor.score(X_train, y_train)
print('Model score:', score)


# In[53]:


evaluate_model(y_test, y_pred)


# ### XGBoost Regressor

# In[54]:


# Building XGBoost model 
# The Extreme gradient boosting decision tree algorithm.
xcb_model = XGBRegressor()

# fit the model with data
xcb_model.fit(X_train, y_train)

# Predict the Xtest data with XGBoost regressor model 
y_pred = xcb_model.predict(X_test)


# In[55]:


# After training the model, we'll check the model training score.

score = xcb_model.score(X_train, y_train)  
print("Model Training score: ", score)


# In[56]:


evaluate_model(y_test, y_pred)


# In[57]:


plt.figure(figsize=(20, 9))
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Loan Lending Club test and predicted data - XGBoost")
plt.legend()
plt.show()


# In[58]:


ax = xgboost.plot_importance(xcb_model, importance_type='weight')
fig = ax.figure
fig.set_size_inches(20, 15)


# In[59]:


ax = xgboost.plot_importance(xcb_model, importance_type='gain')
fig = ax.figure
fig.set_size_inches(20, 15)


# In[60]:


ax = xgboost.plot_importance(xcb_model, importance_type='cover')
fig = ax.figure
fig.set_size_inches(20, 15)


# ### Propose enhancements to the model
# 
# <li> We can tuned the hyper parameters of the model as below
# <li> We can normalized the dataset values in the range (0-1) so all the values of dataset will fall in same unit

# ### Parameterised XGBoost

# In[61]:


from scipy import stats

param_dist = {'n_estimators': stats.randint(550, 1200),
              'learning_rate': stats.uniform(0.01, 0.2),
              'subsample': stats.uniform(0.7, 0.3),
              'max_depth': [5, 6, 7],
              'colsample_bytree': stats.uniform(0.5, 0.5),
              'min_child_weight': [2, 3, 4, 5]
             }

xgbc = XGBRegressor(use_label_encoder=False)
parameterized_xgb = RandomizedSearchCV(
    estimator=xgbc,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    random_state=15,
    n_jobs = -1
)

parameterized_xgb.fit(X_train, y_train)


# In[62]:


pd.DataFrame(parameterized_xgb.cv_results_).sort_values('rank_test_score')


# In[63]:


print(parameterized_xgb.best_params_)


# In[64]:


y_pred = parameterized_xgb.best_estimator_.predict(X_test)


# In[65]:


evaluate_model(y_test, y_pred)


# In[66]:


print("XGBoost Model score: ", parameterized_xgb.best_score_)


# In[67]:


plt.figure(figsize=(20, 9))
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Loan Lending Club test and predicted data - Parameterized XGBoost")
plt.legend()
plt.show()


# ### Conclusion
# <li> Used below algorithms to predict the interest_rest <br>   
# 1) Linear Regression <br>   
# 2) Random Forest Regresoor <br>
# 3) XGBoost Regressor <br>
#     
# <li>By applying all the above models to the Loan Lending Club dataset we are getting good accuracy and R Squared with low Mean Squared Error, Root Mean Squared Error, The Mean Absolute Percentage Error (MAPE) values for XGBoost Model with tuned parameters.
# <li> We also found paid_principle, paid_interest,installment,annual_income_grade_depth_to_income, paid_total, balance_loan_amount,total_debit_limit are important features in predicting the interest_rate more accurately.
# 

# ### Thank you!

# In[ ]:




