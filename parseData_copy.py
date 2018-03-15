import numpy as np
import pandas as pd
import csv
from datetime import datetime
import sklearn
from sklearn.linear_model import LinearRegression

#economy, microsoft, obama, palestine
articles = {}

with open('Data/News_Final.csv', 'r') as csvfile:
    ifile = csv.reader(csvfile)
    next(ifile, None) #skip header
    for line in ifile:
        instance = []
        flag = False
        for val in line:
            if flag:
                instance.append(val)
            else:
                id_link = val
                flag = True
        articles[id_link] = instance

topics = {'economy': [], 'microsoft': [], 'obama': [], 'palestine': []}

for i in articles:
    if articles[i][3] == 'economy':
        topics['economy'].append(i)
    elif articles[i][3] == 'microsoft':
        topics['microsoft'].append(i)
    elif articles[i][3] == 'obama':
        topics['obama'].append(i)
    else:
        topics['palestine'].append(i)

econ_sum_fb = 0
econ_sum_gp = 0
econ_sum_li = 0
econ_count_fb = 0
econ_count_gp = 0
econ_count_li = 0

for i in topics['economy']:
    if float(articles[i][7]) != -1.:
        econ_sum_fb += float(articles[i][7])
        econ_count_fb += 1.
    if float(articles[i][8]) != -1.:
        econ_sum_gp += float(articles[i][8])
        econ_count_gp += 1.
    if float(articles[i][9]) != -1.:
        econ_sum_li += float(articles[i][9])
        econ_count_li += 1.
econ_avg_fb = econ_sum_fb/econ_count_fb
econ_avg_gp = econ_sum_gp/econ_count_gp
econ_avg_li = econ_sum_li/econ_count_li

micro_sum_fb = 0
micro_sum_gp = 0
micro_sum_li = 0
micro_count_fb = 0
micro_count_gp = 0
micro_count_li = 0

for i in topics['microsoft']:
    if float(articles[i][7]) != -1.:
        micro_sum_fb += float(articles[i][7])
        micro_count_fb += 1.
    if float(articles[i][8]) != -1.:
        micro_sum_gp += float(articles[i][8])
        micro_count_gp += 1.
    if float(articles[i][9]) != -1.:
        micro_sum_li += float(articles[i][9])
        micro_count_li += 1.
micro_avg_fb = micro_sum_fb/micro_count_fb
micro_avg_gp = micro_sum_gp/micro_count_gp
micro_avg_li = micro_sum_li/micro_count_li

obama_sum_fb = 0
obama_sum_gp = 0
obama_sum_li = 0
obama_count_fb = 0
obama_count_gp = 0
obama_count_li = 0

for i in topics['obama']:
    if float(articles[i][7]) != -1.:
        obama_sum_fb += float(articles[i][7])
        obama_count_fb += 1.
    if float(articles[i][8]) != -1.:
        obama_sum_gp += float(articles[i][8])
        obama_count_gp += 1.
    if float(articles[i][9]) != -1.:
        obama_sum_li += float(articles[i][9])
        obama_count_li += 1.
obama_avg_fb = obama_sum_fb/obama_count_fb
obama_avg_gp = obama_sum_gp/obama_count_gp
obama_avg_li = obama_sum_li/obama_count_li

pal_sum_fb = 0
pal_sum_gp = 0
pal_sum_li = 0
pal_count_fb = 0
pal_count_gp = 0
pal_count_li = 0

for i in topics['palestine']:
    if float(articles[i][7]) != -1.:
        pal_sum_fb += float(articles[i][7])
        pal_count_fb += 1.
    if float(articles[i][8]) != -1.:
        pal_sum_gp += float(articles[i][8])
        pal_count_gp += 1.
    if float(articles[i][9]) != -1.:
        pal_sum_li += float(articles[i][9])
        pal_count_li += 1.
pal_avg_fb = pal_sum_fb/pal_count_fb
pal_avg_gp = pal_sum_gp/pal_count_gp
pal_avg_li = pal_sum_li/pal_count_li


for i in articles:
    if articles[i][3] == 'economy':
        if float(articles[i][7]) == -1.:
            articles[i][7] = econ_avg_fb
        if float(articles[i][8]) == -1.:
            articles[i][8] = econ_avg_gp
        if float(articles[i][9]) == -1.:
            articles[i][9] = econ_avg_li
    elif articles[i][3] == 'microsoft':
        if float(articles[i][7]) == -1.:
            articles[i][7] = micro_avg_fb
        if float(articles[i][8]) == -1.:
            articles[i][8] = micro_avg_gp
        if float(articles[i][9]) == -1.:
            articles[i][9] = micro_avg_li
    elif articles[i][3] == 'obama':
        if float(articles[i][7]) == -1.:
            articles[i][7] = obama_avg_fb
        if float(articles[i][8]) == -1.:
            articles[i][8] = obama_avg_gp
        if float(articles[i][9]) == -1.:
            articles[i][9] = obama_avg_li
    else:
        if float(articles[i][7]) == -1.:
            articles[i][7] = pal_avg_fb
        if float(articles[i][8]) == -1.:
            articles[i][8] = pal_avg_gp
        if float(articles[i][9]) == -1.:
            articles[i][9] = pal_avg_li

start_date = '1970-01-01 '
for i in articles:
    date = articles[i][4].split(" ")
    date = start_date + date[1]
    datetime_object = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    datetime_object = datetime_object.timestamp() - 28800.
    articles[i][4]=datetime_object

data_init = []
for i in articles:
    temp = []
    fcount = 0
    for val in articles[i]:
        if fcount in (0, 1, 2):
            fcount += 1
            continue
        elif fcount == 3:
            if val == "economy":
                temp.append(1)
                temp.append(0)
                temp.append(0)
                temp.append(0)
            elif val == "microsoft":
                temp.append(0)
                temp.append(1)
                temp.append(0)
                temp.append(0)
            elif val == "obama":
                temp.append(0)
                temp.append(0)
                temp.append(1)
                temp.append(0)
            else:
                temp.append(0)
                temp.append(0)
                temp.append(0)
                temp.append(1)
        else:
            temp.append(float(val))
        fcount += 1
    data_init.append(temp)

'''
print(micro_avg_fb, micro_avg_gp, micro_avg_li)
print(econ_avg_fb, econ_avg_gp, econ_avg_li)
print(obama_avg_fb, obama_avg_gp, obama_avg_li)
print(pal_avg_fb, pal_avg_gp, pal_avg_li)
'''

data = pd.DataFrame(np.matrix(data_init))

data.columns = ['Economy', 'Microsoft', 'Obama', 'Palestine', 'Time (s)', 'SentimentTitle', 'SentimentHeadline', 'Facebook', 'GooglePlus', 'LinkedIn']

X = data.drop(['Facebook', 'GooglePlus', 'LinkedIn'], axis = 1)

Y_fb = data.Facebook
Y_gp = data.GooglePlus
Y_li = data.LinkedIn

TR_X_fb, TE_X_fb, TR_Y_fb, TE_Y_fb = sklearn.model_selection.train_test_split(X, Y_fb, test_size = 0.2, random_state = 5)
TR_X_gp, TE_X_gp, TR_Y_gp, TE_Y_gp = sklearn.model_selection.train_test_split(X, Y_gp, test_size = 0.2, random_state = 5)
TR_X_li, TE_X_li, TR_Y_li, TE_Y_li = sklearn.model_selection.train_test_split(X, Y_li, test_size = 0.2, random_state = 5)


lm1 = LinearRegression()
lm2 = LinearRegression()
lm3 = LinearRegression()
lm1.fit(TR_X_fb, TR_Y_fb)
lm2.fit(TR_X_gp, TR_Y_gp)
lm3.fit(TR_X_li, TR_Y_li)

print ('FB:', pd.DataFrame(list(zip(TR_X_fb.columns, lm1.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('GP:', pd.DataFrame(list(zip(TR_X_gp.columns, lm2.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('LI:', pd.DataFrame(list(zip(TR_X_li.columns, lm3.coef_)), columns = ['Features', 'EstimatedCoefficients']))

error_fb = np.sqrt(np.mean((TE_Y_fb - lm1.predict(TE_X_fb)) ** 2))
error_gp = np.sqrt(np.mean((TE_Y_gp - lm2.predict(TE_X_gp)) ** 2))
error_li = np.sqrt(np.mean((TE_Y_li - lm3.predict(TE_X_li)) ** 2))

print ('FB Error:', error_fb, '\n', 'GP Error:', error_gp, '\n', 'LI Error:', error_li)
