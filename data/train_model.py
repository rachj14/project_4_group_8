import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine
import joblib
from scipy.stats import randint as sp_randint

# Create a connection to the SQLite database
engine = create_engine('sqlite:///heart_disease.db')

# Retrieve data from the SQL database
df = pd.read_sql('SELECT * FROM heart_disease_table', engine)

# Check for missing values and fill them
df.fillna(df.mean(), inplace=True)

# Splitting the dataset into features (X) and target (y)
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# Normalize and standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': sp_randint(100, 200),
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': sp_randint(2, 5),
    'min_samples_leaf': sp_randint(1, 3),
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best parameters and best score from RandomizedSearchCV
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: ", best_score)

# Retrain the model with the best parameters on the full dataset
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_scaled, y)

# Save the best model
joblib.dump(best_model, 'heart_disease_model.pkl')

# Evaluate the best model on the full dataset
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print("Optimised model accuracy on full dataset:", best_accuracy)
print("Classification report for the optimised model:")
print(classification_report(y_test, y_pred_best))

# Document the optimisation process in a CSV file
optimisation_results = pd.DataFrame(random_search.cv_results_)
optimisation_results.to_csv('model_optimisation_results.csv', index=False)

# Plot feature importances
import matplotlib.pyplot as plt

feature_importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('static/feature_importances.png')
plt.show()
