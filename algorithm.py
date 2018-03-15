import numpy as np
import pandas as pd
import csv
from datetime import datetime
import sklearn
from sklearn.linear_model import LinearRegression

def parsedata(filename):
    articles = {}

    with open(filename, 'r') as csvfile:
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

    return articles

def organize_topics(articles):
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

    return topics

def replace_NegOnes(articles, topics):
    econ_avg_fb, econ_avg_gp, econ_avg_li = get_avg_pop(articles, topics, 'economy')
    micro_avg_fb, micro_avg_gp, micro_avg_li = get_avg_pop(articles, topics, 'microsoft')
    obama_avg_fb, obama_avg_gp, obama_avg_li = get_avg_pop(articles, topics, 'obama')
    pal_avg_fb, pal_avg_gp, pal_avg_li = get_avg_pop(articles, topics, 'palestine')

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
    return articles

def get_avg_pop(articles, topics_dict, topic):
    sum_fb = 0
    sum_gp = 0
    sum_li = 0
    count_fb = 0
    count_gp = 0
    count_li = 0

    for i in topics_dict[topic]:
        if float(articles[i][7]) != -1.:
            sum_fb += float(articles[i][7])
            count_fb += 1.
        if float(articles[i][8]) != -1.:
            sum_gp += float(articles[i][8])
            count_gp += 1.
        if float(articles[i][9]) != -1.:
            sum_li += float(articles[i][9])
            count_li += 1.
    avg_fb = sum_fb/count_fb
    avg_gp = sum_gp/count_gp
    avg_li = sum_li/count_li

    return avg_fb, avg_gp, avg_li

def add_seconds_from_date(articles):
    start_date = '1970-01-01 '
    for i in articles:
        date = articles[i][4].split(" ")
        date = start_date + date[1]
        articles[i][4] = get_seconds(date)

def get_seconds(date):
    datetime_object = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return datetime_object.timestamp() - 28800.

def create_dataframe(articles):
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

    return data

def prepare_data(data):
    X = data.drop(['Facebook', 'GooglePlus', 'LinkedIn'], axis = 1)
    Y_fb = data.Facebook
    Y_gp = data.GooglePlus
    Y_li = data.LinkedIn

    data_dict = {'X':  {'TR': {'fb': 1, 'gp': 1, 'li': 1},
                        'TE': {'fb': 1, 'gp': 1, 'li': 1}} ,
                 'Y': {'TR': {'fb': 1, 'gp': 1, 'li': 1},
                       'TE': {'fb': 1, 'gp': 1, 'li': 1}}}

    data_dict['X']['TR']['fb'], data_dict['X']['TE']['fb'], data_dict['Y']['TR']['fb'], data_dict['Y']['TE']['fb'] = sklearn.model_selection.train_test_split(X, Y_fb, test_size = 0.2, random_state = 5)
    data_dict['X']['TR']['gp'], data_dict['X']['TE']['gp'], data_dict['Y']['TR']['gp'], data_dict['Y']['TE']['gp'] = sklearn.model_selection.train_test_split(X, Y_gp, test_size = 0.2, random_state = 5)
    data_dict['X']['TR']['li'], data_dict['X']['TE']['li'], data_dict['Y']['TR']['li'], data_dict['Y']['TE']['li'] = sklearn.model_selection.train_test_split(X, Y_li, test_size = 0.2, random_state = 5)

    return data_dict

def linear_regression(data_dict):
    lm1 = LinearRegression()
    lm2 = LinearRegression()
    lm3 = LinearRegression()

    lm1.fit(data_dict['X']['TR']['fb'], data_dict['Y']['TR']['fb'])
    lm2.fit(data_dict['X']['TR']['gp'], data_dict['Y']['TR']['gp'])
    lm3.fit(data_dict['X']['TR']['li'], data_dict['Y']['TR']['li'])

    print ('FB:', pd.DataFrame(list(zip(data_dict['X']['TR']['fb'].columns, lm1.coef_)), columns = ['Features', 'EstimatedCoefficients']))
    print ('GP:', pd.DataFrame(list(zip(data_dict['X']['TR']['gp'].columns, lm2.coef_)), columns = ['Features', 'EstimatedCoefficients']))
    print ('LI:', pd.DataFrame(list(zip(data_dict['X']['TR']['li'].columns, lm3.coef_)), columns = ['Features', 'EstimatedCoefficients']))

    error_fb = np.sqrt(np.mean((data_dict['Y']['TE']['fb'] - lm1.predict(data_dict['X']['TE']['fb'])) ** 2))
    error_gp = np.sqrt(np.mean((data_dict['Y']['TE']['gp'] - lm2.predict(data_dict['X']['TE']['gp'])) ** 2))
    error_li = np.sqrt(np.mean((data_dict['Y']['TE']['li'] - lm3.predict(data_dict['X']['TE']['li'])) ** 2))

    print ('FB Error:', error_fb, '\nGP Error:', error_gp, '\nLI Error:', error_li)

if __name__ == "__main__":
    articles = parsedata('Data/News_Final.csv')
    topics = organize_topics(articles)
    replace_NegOnes(articles, topics)
    add_seconds_from_date(articles)
    data = create_dataframe(articles)
    data_dict = prepare_data(data)
    linear_regression(data_dict)
