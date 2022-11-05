#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv(".../data.csv").drop('Unnamed: 0', axis=1)
X = df.drop('model...', axis=1)
y = df['model...']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 75)


#for_classification:
# Note: for regression problems, use XGBRegressor()

from xgboost import XGBClassifier
# please use parameters from the table 4 in the paper 
params = {
            'n_estimators':'binary:logistic':500 ,
            'max_depth':6 ,
            'Subsample':0.65,
            'Colsample_bytree':0.4,
            'learning_rate':0.01,
            'alpha':0.25 ,
            'lambda':1.3 
        }
            
            
            
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

# alternatively view the parameters of the xgb trained model
print(xgb_clf)


# In[ ]:


# make predictions on test data
y_pred = xgb_clf.predict(X_test)


# In[ ]:


import shap

# explain the model's predictions using SHAP
# please choose suitable arguments for each model

X_sampled = X_train.sample(100, random_state=10)
explainer = shap.KernelExplainer(xgb_clf, X_train)
shap_values = explainer.shap_values(X_sampled)
shap.summary_plot(shap_values, X_sampled)


# In[ ]:




