'''
pip install plotly
'''

import plotly.express as px

# Example: Interactive scatter plot
df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
fig.show()
