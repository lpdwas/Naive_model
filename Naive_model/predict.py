from os import name
import pickle
from pretreatment import spilcing,market
from transform import Onehot_encode
import datetime
import numpy as np
import pandas as pd

def date_change(str_date, n_day):
    time_date = datetime.datetime.strptime(str_date,'%Y/%m/%d')
    delta = datetime.timedelta(days=n_day)
    n_days = time_date + delta
    future_days = n_days.strftime('%Y/%#m/%#d')
    return future_days

def test_predict(Naive_model, datapath, textpath, modelname, mindf, fee): #用于直接输出预测结果
    result = []
    with open(Naive_model,'rb') as f:
        clf = pickle.load(f)
    text_data = spilcing(textpath)
    market_data = market(datapath,fee)
    Onehot_encode(text_data,modelname,mindf)
    for day in text_data.keys():
        x = np.array(text_data[day])
        direction = clf.predict(x)
        result.append([day,list(direction)[0]])
    result = pd.DataFrame(result,index=None,columns=['day','pre_direction'])
    data = pd.merge(market_data,result,on='day')
    return data

if __name__ == '__main__':
    name = '动力煤'
    data = test_predict('model/Naivemodel_'+name+'_1.pickle','data/source/'+name+'data.csv','data/text','model/'+name+'_One_hot_1.pickle',round(0.3,2),fee=100)
    print(data)
        
            
            