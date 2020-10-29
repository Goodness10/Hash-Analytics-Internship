#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.ExcelFile('C:/Users/USER/Hash-Analytic-Python-Analytics-Problem-case-study-1 (1).xlsx')
who_left = dataset.parse('Employees who have left')

# To get the correlation betweem attributes
corr_matrix = who_left.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True)
plt.savefig('Employees who have left Correlation')
plt.show()


#Visualising some features
features=['number_project','time_spend_company','Work_accident','promotion_last_5years','dept','salary']
subplot = plt.subplots(figsize=(20,15))
for i,j in enumerate(features):
    plt.subplot(4,2,i+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x=j, data=who_left)
    plt.xticks(rotation=90)
    plt.title('Employees who have left breakdown')
    plt.savefig('Employees who have left breakdown')

