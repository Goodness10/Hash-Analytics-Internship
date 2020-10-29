import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_excel('C:/Users/USER/Hash Analytics Dataset.xlsx')

#Visualising some features
features=['number_project','time_spend_company','Work_accident','promotion_last_5years','dept','salary']
subplot = plt.subplots(figsize=(20,15))
for i,j in enumerate(features):
    plt.subplot(4,2,i+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x=j, data=dataset, hue='status')
    plt.xticks(rotation=90)
    plt.title('Total employees')
    plt.savefig('Total employees histogram')


