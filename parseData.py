import numpy as np
import pandas as pd
import csv
from datetime import datetime

#economy, microsoft, obama, palestine
articles = {}
data_init = []

with open('Data/News_Final.csv', 'r') as csvfile:
    ifile = csv.reader(csvfile)
    #skip header
    next(ifile, None)

    for line in ifile:
        temp = []
        instance = []
        fcount = 0
        for val in line:
            if fcount == 0:
                id_link = val
                fcount += 1
                continue
            instance.append(val)
            if fcount in (1, 2, 3):
                fcount += 1
                continue
            elif fcount == 4:
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
                temp.append(val)
            fcount += 1
        articles[id_link] = instance
        data_init.append(temp)

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

data_init = np.matrix(data_init)


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
    datetime_object = datetime_object.timestamp()
    articles[i][4]=datetime_object
'''
print econ_avg_fb, econ_avg_gp, econ_avg_li
print micro_avg_fb, micro_avg_gp, micro_avg_li
print obama_avg_fb, obama_avg_gp, obama_avg_li
print pal_avg_fb, pal_avg_gp, pal_avg_li
'''
print(articles['1'])
