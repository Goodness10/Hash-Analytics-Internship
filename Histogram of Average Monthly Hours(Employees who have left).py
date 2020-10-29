#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.ExcelFile('C:/Users/USER/Hash-Analytic-Python-Analytics-Problem-case-study-1 (1).xlsx')
who_left = dataset.parse('Employees who have left')

a = who_left.iloc[:, 4].values
plt.hist(a,bins=7,histtype='barstacked')
plt.ylabel('Count')
plt.savefig('Histogram of Average Monthly Hours(Employees who have left)')
plt.show()