# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:39:34 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime


column_name = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'class']

df = pd.read_csv('https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/iris/iris.data.csv', names=column_name)

#Scatterplott, sepal width by length
plt.scatter(df.sepal_l, df.sepal_w)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

#Pick a point at random
random.seed(datetime.now())

rand_sepal_l = random.uniform(df.sepal_l.min(), df.sepal_l.max())
rand_sepal_w = random.uniform(df.sepal_w.min(), df.sepal_w.max())

plt.scatter(rand_sepal_l, rand_sepal_w,color='red')

#Sort each point by its distance from the new point, and subset the 10 nearest points.
#First, we normalize the distance for both variables


df['sepal_l_d'] = abs(df.sepal_l - rand_sepal_l)
df['n_sepal_l_d'] = (df.sepal_l_d - df.sepal_l_d.min()) / (df.sepal_l_d.max() - df.sepal_l_d.min())

df['sepal_w_d'] = abs(df.sepal_w - rand_sepal_w)
df['n_sepal_w_d'] = (df.sepal_w_d - df.sepal_w_d.min()) / (df.sepal_w_d.max() - df.sepal_w_d.min())


#calculate the distance, weighing both variables equally, and keeping the 10 closest points
df['distance'] = df[["n_sepal_l_d", "n_sepal_w_d"]].mean(axis=1)


def knn(n):
    result = df.sort(['distance'], ascending=[1]).head(n)
    freq_results = result['class'].value_counts()
    print(freq_results.index[0])
    print(freq_results)
    
    """Runs well, but produces this message: __main__:47: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=...."""
    
