import pandas as pd
from sqlalchemy import create_engine

# Load the dataset
df = pd.read_csv('heart_disease.csv')

# Drop the unnecessary columns
df = df.drop(columns=['AnyHealthcare', 'NoDocbcCost', 'Income'])

# Create a connection to the SQLite database
engine = create_engine('sqlite:///heart_disease.db')

# Write the data to the SQLite database
df.to_sql('heart_disease_table', engine, if_exists='replace', index=False)

print("Database and table created successfully with required columns.")