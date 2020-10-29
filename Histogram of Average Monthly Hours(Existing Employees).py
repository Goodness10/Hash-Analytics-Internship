#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.ExcelFile('C:/Users/USER/Hash-Analytic-Python-Analytics-Problem-case-study-1 (1).xlsx')
existing_employees = dataset.parse('Existing employees')

a = existing_employees.iloc[:, 4].values
plt.hist(a,bins=7,histtype='barstacked')
plt.ylabel('Count')
plt.savefig('Histogram of Average Monthly Hours(Existing Employees)')
plt.show()