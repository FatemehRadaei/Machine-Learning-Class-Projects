
from sklearn.preprocessing import StandardScaler
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import dataframe as df
name = '../dqn/akhari/features.csv'

with open(name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    value = 0
    list = []
    data = []
    for row in csv_reader:
        for item in row:
            value = re.findall("\d+\.\d+", item)
            if len(value) == 0:
                list.append(0)
            else:
                list.append(float(value[0]))

        data.append(list)
        list = []
x = data
#print(x[0])
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])

color = []
with open('../dqn/akhari/actions.csv') as csv_file:
    for line in csv_file:
        color.append(float(line))


#import IPython; IPython.embed()
x = principalDf.pc1
y = principalDf.pc2
plt.scatter(x,y,c=color)
plt.show()
# change color by award! at each feature! or filter ?
# show loss !
