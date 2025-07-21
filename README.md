# rainfall-predictor--ML
Objective:
The primary goal of this project is to develop a machine learning classifier that can accurately predict whether it will rain on a given day based on historical weather data from the Melbourne area. This is a binary classification task using features such as temperature, humidity, wind direction, and pressure.

 Dataset:
Source: Kaggle – Australian historical weather data

Focused Locations: Melbourne, Melbourne Airport, Watsonia

Target variable: RainToday (Yes / No)

 Methodology:
Data Preparation:

Filtered data for the Melbourne area

Dropped rows with missing values

Performed feature engineering (e.g., Month, Season)

Separated features into numerical and categorical

Modeling:

Built a preprocessing pipeline using ColumnTransformer

Applied SimpleImputer, StandardScaler, and OneHotEncoder

Trained two models:

Logistic Regression (baseline)

Random Forest Classifier (advanced)

Evaluation:

Used stratified train-test split

Compared models using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Visualized feature importance

 Results:
Random Forest Classifier outperformed Logistic Regression:

Accuracy: 87% (vs 84%)

Recall for rainy days (Yes): 57% (vs 51%)

Most important predictive feature: Humidity3pm

Conclusion:
The Random Forest model proved more effective at identifying rainy days than Logistic Regression. This project highlights the importance of proper preprocessing, model evaluation beyond accuracy, and identifying impactful weather features for practical forecasting applications.
