import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
def judge(result,datapath,fee):
    date = result['day']
    result = result.set_index('day')
    data = read_csv(datapath)
    data = data.set_index('day')
    x = []
    y = []
    money = 0
    for i in date:
        openprice = data.loc[i]['openprice']
        closeprice = data.loc[i]['closeprice']
        judgement = result.loc[i]['result']
        if judgement == 1:
            got = (closeprice-openprice)*fee
            money = money +got
            x.append(got)
        elif judgement == 0:
            got = (openprice-closeprice)*fee
            money = money + got 
            x.append(got)
        else:
            money = money
        ave = np.mean(x)
        stt = np.std(x)
        shape_rate = round(ave/stt*15.8,2)
    return money,shape_rate