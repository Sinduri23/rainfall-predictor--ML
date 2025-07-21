#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# In[63]:


##The dataset contains observations of weather metrics for each day from 2008 to 201


# In[2]:


data=pd.read_csv("C:\\Users\\sindu\\Downloads\\weatherAUS.csv\\weatherAUS.csv")


# In[3]:


data.head()


# In[4]:


data.count()


# In[5]:


data.shape


#  We try to keep things simple and drop rows with missing values and see what's left

# In[ ]:


data.dropna()


# In[6]:


data.info()


# In[7]:


data.columns


# If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday, then we can legitimately utilize all of the available features. This shift would be particularly useful for practical applications, such as deciding whether you will bike to work today.
# 
# With this new target, we should update the names of the rain columns accordingly to avoid confusion.

# In[8]:


data=data.rename(columns={'RainToday':'Rainyesterday', 'RainTomorrow':'Raintoday'})


# Watsonia is only 15 km from Melbourne, and the Melbourne Airport is only 18 km from Melbourne.
# We can  group these three locations together and use only their weather data to build our localized prediction model.

# In[9]:


data=data[data.Location.isin(['Melbourne','MelbourneAirport','Watsonia']) ]


# In[10]:


data.info()


# In[11]:


data['Date']=pd.to_datetime(data['Date'])


# In[12]:


data.head()


# In[13]:


data['Month']=data['Date'].dt.month


# In[14]:


data['Month']


# Consider the Date column. We expect the weather patterns to be seasonal, having different predictablitiy levels in winter and summer for example.
# We can engineer a Season feature from Date and drop Date afterward, since it is most likely less informative than season

# In[15]:


def month_season(Month):
    if (Month == 12) or (Month == 1) or (Month == 2):
        return 'Summer'
    elif (Month == 3) or (Month == 4) or (Month == 5):
        return 'Autumn'
    elif (Month == 6) or (Month == 7) or (Month == 8):
        return 'Winter'
    elif (Month == 9) or (Month == 10) or (Month == 11):
        return 'Spring'


# In[16]:


data['Season']=data['Month'].apply(month_season)


# In[17]:


data.drop(columns=['Date','Month'])


# Define feature and target dataframes
# 

# In[25]:


X=data.drop(columns=['Raintoday'])
y=data['Raintoday']


# In[21]:


X.value_counts()


# In[19]:


y.value_counts()


# ## conclusions:
# It rains approximately 1 in 5 days.
# If your classifier always predicted ‘No rain today’, it would be ~80% accurate.
# his is not a balanced dataset.
# It is imbalanced, with a strong skew toward “no rain.”
# 

# In[26]:


print("Missing in X:", X.isna().sum().sum())
print("Missing in y:", y.isna().sum())
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[27]:


# Step 1: Create a mask for non-missing values in y
mask = ~y.isna()

# Step 2: Apply the mask to both X and y
X = X[mask]
y = y[mask]


# ## Split data into training and test sets, ensuring target stratification

# In[28]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


# ## Define preprocessing transformers for numerical and categorical features

# In[30]:


numerical_features=X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features=X_train.select_dtypes(include=['object','category']).columns.tolist()                                         


# In[32]:


from sklearn.impute import SimpleImputer


# ## Define separate transformers for both feature types and combine them into a single preprocessing transformer

# In[33]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
                                      


# ## Combine the transformers into a single preprocessing column transformer

# In[34]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# ## Create a pipeline by combining the preprocessing with a Random Forest classifier

# In[35]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# ## Define a parameter grid to use in a cross validation grid search model optimizer

# In[36]:


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}


# ## Define a parameter grid to use in a cross validation grid search model optimizer

# In[37]:


# Cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True)


# In[38]:


grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)


# In[40]:


# Enter your code here:
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))


# In[41]:


# Enter your code here:
conf_matrix =confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('rainfall prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# In[42]:


grid_search.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)


# In[45]:


print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[46]:


# Print test score 
test_score = grid_search.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")


# we have a reasonably accurate classifier, which is expected to correctly predict about 86% of the time whether it will rain today in the Melbourne area.

# In[43]:


feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))


# In[ ]:





# In[50]:


# Combine numeric and categorical feature names
feature_names = numerical_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


# From  RandomForestClassifier feature importances, the top 20 plot, the feature  with  the highest importance score is:
# 
# Humidity3pm

# ## another model
# 
# we can try different models to improve model's performance

# In[51]:


pipeline.set_params(classifier=LogisticRegression(random_state=42))


# In[55]:


grid_search.estimator=pipeline


# In[56]:


param_grid={'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']}
    


# In[57]:


grid_search.param_grid=param_grid


# In[58]:


grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)


# In[67]:


print(classification_report(y_test, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title(' Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# In[60]:


coefficients = grid_search.best_estimator_.named_steps['classifier'].coef_[0]

# Combine numerical and categorical feature names
numerical_feature_names = numerical_features
categorical_feature_names = (grid_search.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = numerical_feature_names + list(categorical_feature_names)


# In[62]:


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values
N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)
# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = grid_search.best_estimator_.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")


# ## Comparison of RandomForestClassifier vs LogisticRegression
# 1. Accuracy
# RandomForest: 0.87
# 
# LogisticRegression: 0.84
# 
#  Conclusion: RandomForest performs better overall.
# 
# 2. True Positive Rate (Recall for "Yes" = Rain)
# RandomForest: 0.57
# 
# LogisticRegression: 0.51
# 
#  Conclusion: RandomForest is slightly better at correctly identifying rainy days.
# 
# 3. Precision for "Yes" (Rain)
# RandomForest: 0.82
# 
# LogisticRegression: 0.72
# 
#  Conclusion: RandomForest is more confident when predicting "Yes" — fewer false positives.
# 
# 4. F1-Score for "Yes"
# RandomForest: 0.67
# 
# LogisticRegression: 0.60
# 
#  Conclusion: RandomForest has a better balance between precision and recall for rain prediction.
# 
# 5. Macro and Weighted Averages
# Macro F1 (treats both classes equally):
# 
# RandomForest: 0.80
# 
# LogisticRegression: 0.75
# 
# Weighted F1 (accounts for class imbalance):
# 
# RandomForest: 0.86
# 
# LogisticRegression: 0.82
# 
#  RandomForest handles both metrics better.
# 
# 

# ## Conclusion Summary
#  RandomForestClassifier outperforms LogisticRegression in all key metrics: accuracy, precision, recall, and F1-score.
# 
#  RandomForest is more effective at predicting rainfall (Yes), which is critical for this classification task.
# 
#  LogisticRegression is simpler but underperforms, especially in identifying rainy days (lower recall and F1).
# 
#  If predicting rain accurately is important, RandomForest is the better model for deployment

# In[ ]:




