'''
install the below packages through command prompt
pip install matplotlib
pip install seaborn
'''

import matplotlib.pyplot as plt
import seaborn as sns

# Example: Basic data visualization
sns.set(style="whitegrid")
data = [1, 5, 3, 6, 4, 2]
plt.plot(data)
plt.title("Line Graph")
plt.show()

# Matplotlib provides plotting functionality, and Seaborn enhances this with a high-level interface and attractive default styles.