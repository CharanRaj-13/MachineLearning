'''
pip install pandas
'''

import pandas as pd

# Example: Creating and manipulating a DataFrame
# For Data Manipulation
data = {'Name': ['Alice', 'Bob', 'Nick'], 'Age': [25, 30, 19]}
df = pd.DataFrame(data)
print(df)
print("Average Age:", df['Age'].mean())

# Used for data preprocessing, exploration, and manipulation with DataFrames and Series.