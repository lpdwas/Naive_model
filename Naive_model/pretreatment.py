#--encoding-- = utf-8
import jieba
import os 
import re
import datetime
import pandas as pd

#主要用于文本预处理，包括jieba切词以及市场价格的读取及滤除

def spilcing(filepath): 
    stopwords = open('data\stopwords.txt','r',encoding='utf-8-sig').read().split()
    filenames = os.listdir(filepath)
    pattern = re.compile('\d{4}\d{2}\d{2}')
    din_word = []
    source_data = {}
    for filename in filenames:
        date = pattern.findall(filename)[0]
        date = datetime.datetime.strptime(date,'%Y%m%d').strftime('%Y/%#m/%#d')
        data = open(filepath + '/' + filename,'r',encoding='utf-8-sig').read().replace('\n','')
        words = []
        items = jieba.lcut(data)
        for n in range(len(items)-1,-1,-1):  
            word= items[n]
            if word in stopwords:
                items.pop(n)
                continue
            if '.' in word:
                items.pop(n)
                continue
            if '\uf06c' in word:
                items.pop(n)
                continue
            for ky in word :
                if ky in '0 1 2 3 4 5 6 7 8 9 （ ） 《 》 / 【 】 . -：• '.split():
                    items.pop(n)
                    break
            for k in items:
                words.append(k)
                if k in '。 ， ？ ！'.split():
                    words.remove(k)
                    din_word.append(words)
                    words =[]
            try :
                source_data[date]
            except:
                source_data[date] = []
            source_data[date].append(din_word)
            din_word = []
        print('当前进度为:转换{}日'.format(date),end='\r')
    return source_data
def market(datapath,fee):
    data = pd.read_csv(datapath, encoding='utf-8')
    date = data['day']
    data = data.set_index('day')
    market = []
    for time in date:
        openprice = data.loc[time]['openprice']
        closeprice = data.loc[time]['closeprice']
        opentime = data.loc[time]['opentime']
        closetime = data.loc[time]['closetime']
        PNL = (closeprice - openprice)*fee
        price_return = round(PNL/(fee*closeprice),2)
        judge = (closeprice-openprice)/closeprice
        if judge >= 0.008:
            state = 1
        elif judge < 0:
            state = 0
        else:
            continue
        market.append([time,opentime,openprice,closetime,closeprice,state,PNL,price_return])
    market = pd.DataFrame(market,index=None,columns=['day','opentime','openprice','closetime','closeprice','realdirection','PNL','return'])
    return market  # 涨跌情况存放到以日期为键的字典当中
if __name__ == '__main__':
    data = spilcing('train_data\文本')
    print(data)