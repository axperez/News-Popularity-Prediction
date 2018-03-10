import numpy as np
import pandas as pd
import csv

#economy, microsoft, obama, palestine

data_init = []

with open('Data/News_Final.csv', 'rb') as csvfile:
    ifile = csv.reader(csvfile)
    next(ifile, None) #skip header

    for line in ifile:
        temp = []
        fcount = 0
        for val in line:
            if fcount in (0, 1, 2, 3):
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
        data_init.append(temp)

data_init = np.matrix(data_init)
print data_init
