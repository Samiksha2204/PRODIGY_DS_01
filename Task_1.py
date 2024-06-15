import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Data.csv", delimiter=',', quotechar='"', skiprows=0, on_bad_lines='skip')
print("Initial DataFrame shape:", df.shape)

# Handling Missing Values
df = df.dropna()
print("After dropping NaNs:", df.shape)

# Removing Duplicates
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

#Data Type Conversion
# df['date_column'] = pd.to_datetime(df['date_column'])
print("Data types after conversion:\n", df.dtypes)

#Handling Outliers
#Remove outliers using the IQR method
if not df.empty:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("After removing outliers:", df.shape)

#Standardizing and Normalizing Data
if not df.empty:
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
print("After scaling:\n", df.head())

#Handling Categorical Data
# Convert categorical columns to numerical using one-hot encoding 
print("Final DataFrame shape:", df.shape)
print("Data types:\n", df.dtypes)
print(df.head())

# Visualization of Gender Distribution (Bar Chart)
data = {
    'gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'age': [22, 25, 29, 34, 45, 33, 27, 25, 31, 40]
}
df_vis = pd.DataFrame(data)  # Creating a new DataFrame for visualization purposes

gender_counts = df_vis['gender'].value_counts()

plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Distribution of Genders')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Histogram for Age Distribution
plt.figure(figsize=(8, 6))
plt.hist(df_vis['age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
