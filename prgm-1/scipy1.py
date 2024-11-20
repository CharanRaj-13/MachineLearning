'''
pip install scipy
pip install numpy

'''

from scipy import stats
import numpy as np

# Example: Statistical analysis
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)
std_dev = np.std(data)
t_stat, p_val = stats.ttest_1samp(data, 3)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
print(f"T-Statistic: {t_stat}, P-Value: {p_val}")
# Useful for advanced mathematics, scientific calculations, and performing statistical tests such as T-tests, ANOVA, and linear algebra.