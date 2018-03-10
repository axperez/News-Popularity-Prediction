import pandas as pd
import sklearn
import csv

def parsedata(filename):
    sources = {}
    comm_srcs = []
    with open(filename, "rb") as csvfile:
        ifile = csv.reader(csvfile)
        next(ifile, None)
        for line in ifile:
            fcount = 0
            for value in line:
                fcount += 1
                if fcount == 4:
                    if value in sources.keys():
                        sources[value][0] += 1
                        if sources[value][0] > 1000 and sources[value][1] == 0:
                            comm_srcs.append(value)
                            sources[value][1] = 1
                    else:
                        sources[value] = [1, 0]
    #print sources
    print len(sources)
    print comm_srcs
    print len(comm_srcs)
    return



if __name__ == "__main__":
    parsedata("/Users/axelperez97/Documents/SCU/Machine_Learning/News-Popularity-Prediction/Data/News_Final.csv")
