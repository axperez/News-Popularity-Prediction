import numpy as np
import pandas as pd
import csv
from datetime import datetime
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
#economy, microsoft, obama, palestine

#########################################################################################################
#PARSING DATA

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

data = pd.DataFrame(np.matrix(data_init))

data.columns = ['Economy', 'Microsoft', 'Obama', 'Palestine', 'Time (s)', 'SentimentTitle', 'SentimentHeadline', 'Facebook', 'GooglePlus', 'LinkedIn']

X = data.drop(['Facebook', 'GooglePlus', 'LinkedIn'], axis = 1)

Y_fb = data.Facebook
Y_gp = data.GooglePlus
Y_li = data.LinkedIn

TR_X_fb, TE_X_fb, TR_Y_fb, TE_Y_fb = sklearn.model_selection.train_test_split(X, Y_fb, test_size = 0.2, random_state = 5)
TR_X_gp, TE_X_gp, TR_Y_gp, TE_Y_gp = sklearn.model_selection.train_test_split(X, Y_gp, test_size = 0.2, random_state = 5)
TR_X_li, TE_X_li, TR_Y_li, TE_Y_li = sklearn.model_selection.train_test_split(X, Y_li, test_size = 0.2, random_state = 5)

#########################################################################################################
#LINEAR REGRESSION

print('Linear Regression Model:')
lm1 = LinearRegression()
lm2 = LinearRegression()
lm3 = LinearRegression()
lm1.fit(TR_X_fb, TR_Y_fb)
lm2.fit(TR_X_gp, TR_Y_gp)
lm3.fit(TR_X_li, TR_Y_li)

print ('FB:', pd.DataFrame(list(zip(TR_X_fb.columns, lm1.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('GP:', pd.DataFrame(list(zip(TR_X_gp.columns, lm2.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('LI:', pd.DataFrame(list(zip(TR_X_li.columns, lm3.coef_)), columns = ['Features', 'EstimatedCoefficients']))

error_fb = np.mean((TE_Y_fb - lm1.predict(TE_X_fb)) ** 2)
error_gp = np.mean((TE_Y_gp - lm2.predict(TE_X_gp)) ** 2)
error_li = np.mean((TE_Y_li - lm3.predict(TE_X_li)) ** 2)

print ('FB Error:', error_fb, '\n', 'GP Error:', error_gp, '\n', 'LI Error:', error_li)
print('\n')
#########################################################################################################
#RIDGE REGRESSION
print('Ridge Regression Model:')
reg1 = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 20.0, 30.0, 50.0, 70.0, 85.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 150.0, 175.0, 200.0])
reg2 = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 20.0, 30.0, 50.0, 70.0, 85.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 200.0])
reg3 = linear_model.RidgeCV(alphas=[110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 150.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 275.0])

reg1.fit(TR_X_fb, TR_Y_fb)
reg2.fit(TR_X_gp, TR_Y_gp)
reg3.fit(TR_X_li, TR_Y_li)

#print(reg1.alpha_) #125
#print(reg2.alpha_) #170
#print(reg3.alpha_) #265

reg1 = linear_model.Ridge (alpha = reg1.alpha_)
reg2 = linear_model.Ridge (alpha = reg2.alpha_)
reg3 = linear_model.Ridge (alpha = reg3.alpha_)

reg1.fit(TR_X_fb, TR_Y_fb)
reg2.fit(TR_X_gp, TR_Y_gp)
reg3.fit(TR_X_li, TR_Y_li)

print ('FB:', pd.DataFrame(list(zip(TR_X_fb.columns, reg1.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('GP:', pd.DataFrame(list(zip(TR_X_gp.columns, reg2.coef_)), columns = ['Features', 'EstimatedCoefficients']))
print ('LI:', pd.DataFrame(list(zip(TR_X_li.columns, reg3.coef_)), columns = ['Features', 'EstimatedCoefficients']))

error_fb = np.mean((TE_Y_fb - reg1.predict(TE_X_fb)) ** 2)
error_gp = np.mean((TE_Y_gp - reg3.predict(TE_X_gp)) ** 2)
error_li = np.mean((TE_Y_li - reg3.predict(TE_X_li)) ** 2)

print ('FB Error:', error_fb, '\n', 'GP Error:', error_gp, '\n', 'LI Error:', error_li)
print('\n')

#########################################################################################################
#RANDOM FOREST REGRESSOR
print('Random Forest Regressor:')

regr1 = RandomForestRegressor(n_estimators=7)
regr2 = RandomForestRegressor(n_estimators=7)
regr3 = RandomForestRegressor(n_estimators=7)

regr1.fit(TR_X_fb, TR_Y_fb)
regr2.fit(TR_X_gp, TR_Y_gp)
regr3.fit(TR_X_li, TR_Y_li)

print ('FB:', pd.DataFrame(list(zip(TR_X_fb.columns, regr1.feature_importances_)), columns = ['Features', 'EstimatedFeatureImportances']))
print ('GP:', pd.DataFrame(list(zip(TR_X_gp.columns, regr2.feature_importances_)), columns = ['Features', 'EstimatedFeatureImportances']))
print ('LI:', pd.DataFrame(list(zip(TR_X_li.columns, regr3.feature_importances_)), columns = ['Features', 'EstimatedFeatureImportances']))

error_fb = np.mean((TE_Y_fb - reg1.predict(TE_X_fb)) ** 2)
error_gp = np.mean((TE_Y_gp - regr2.predict(TE_X_gp)) ** 2)
error_li = np.mean((TE_Y_li - regr3.predict(TE_X_li)) ** 2)

print ('FB Error:', error_fb, '\n', 'GP Error:', error_gp, '\n', 'LI Error:', error_li)
