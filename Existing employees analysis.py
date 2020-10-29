#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.ExcelFile('C:/Users/USER/Hash-Analytic-Python-Analytics-Problem-case-study-1 (1).xlsx')
existing_employees = dataset.parse('Existing employees')



# To get the correlation betweem attributes
corr_matrix = existing_employees.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True)
plt.savefig('Existing Employees Correlation.png')
plt.show()

#Visualising some features
features=['number_project','time_spend_company','Work_accident','promotion_last_5years','dept','salary']
subplot = plt.subplots(figsize=(20,15))
for i,j in enumerate(features):
    plt.subplot(4,2,i+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x=j, data=existing_employees)
    plt.xticks(rotation=90)
    plt.title('Existing Employees')
    plt.savefig('Existing Employees')

