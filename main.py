import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle
import joblib
# Load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# List of columns to drop
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']

# Drop the specified columns
dataset = dataset.drop(columns=columns_to_drop)

# Define features and target
X = dataset.drop('Exited', axis=1)
y = dataset['Exited']

# Define numerical and categorical features
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                      'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Gradient Boosting model
model = GradientBoostingClassifier()

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f'Gradient Boosting Classifier: {scores.mean():.4f} (+/- {scores.std():.4f})')

# Predict on the test set
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the pipeline
joblib.dump(pipeline, 'pipeline.joblib')


# Save the DataFrame
with open('df.pkl', 'wb') as file:
    pickle.dump(X, file)


