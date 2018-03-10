
# coding: utf-8

# In[19]:

import numpy as np
import pandas as pd
import csv


# # load into matrix 

# In[23]:

data_init = []

with open('News_Final.csv', 'r') as csvfile:
    ifile = csv.reader(csvfile)
    next(ifile, None) #skip header
    
    for line in ifile:
        for val in line:
            data_init.append(val)
            
data_init = np.matrix(data_init)
data_init = data_init.reshape(93239,11)
#print(data_init) #looks good!



# # add columns for topic classification

# In[31]:

N = 93239
all_data = []
all_data = np.c_[data_init,np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)]
#now four columns of zeros appended to matrix 

#topic is in 4th column (starting with zero)
for row in all_data:
    print(all_data[row,4])
       

