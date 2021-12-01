from sklearn.naive_bayes import BernoulliNB
import pickle
import numpy as np
import datetime
import pandas as pd
#进行时间格式变换
def date_change(str_date, n_day):
    time_date = datetime.datetime.strptime(str_date,'%Y/%m/%d')
    delta = datetime.timedelta(days=n_day)
    n_days = time_date + delta
    future_days = n_days.strftime('%Y/%#m/%#d')
    return future_days

#进行Naive Bayes模型训练
def Naive (X_din,Y_din,Naive_model_path,n_day):
    clf = BernoulliNB()
    X_list = []
    Y_list = []
    offset = int(len(Y_din.keys())*0.7)  #训练集比例
    for key in list(Y_din.keys())[:offset]: 
        new_key = date_change(key, n_day)
        if key in X_din and new_key in Y_din:  # 通过extend函数将词汇和涨跌表现分别整合
            X_list.extend(X_din[key])
            Y_list.extend([Y_din[new_key]])
    Y = np.array(Y_list)
    X = np.array(X_list)
    clf.fit(X,Y)
    right = clf.score(X,Y)  # 输出拟合的准确度
    with open(Naive_model_path,'wb') as f:  # 对训练出来的模型进行储存
        pickle.dump(clf,f) 
    return offset,right

def predict_data(new_din, Y_din, Naive_model_path, n_day, offset, state, stage, name):  #对训练集测试集进行预测
    y_pre = []
    y_label = []
    predict = []
    real_predict = []
    all_amo = 0
    n = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if stage:
        ain = list(Y_din.keys())[:offset] #训练集
    else:
        ain = list(Y_din.keys())[offset:] #测试集
    with open (Naive_model_path,'rb') as f:
        clf = pickle.load(f)
    for key in ain:
        new_key = date_change(key, n_day)
        if key not in new_din.keys() and new_key in Y_din.keys():
            real_predict.append([None,new_key,key])
        elif key in new_din.keys() and new_key in Y_din.keys(): 
            all_amo += 1
            X = np.array(new_din[key])
            a = clf.predict(X)
            b = list(a)
            b.append(new_key)
            b.append(key)
            real_predict.append(b)
            if a == Y_din[new_key]:
                n += 1
                try:
                    predict = np.array(predict)
                    predict = predict + X 
                except:
                    predict = X
                if list(a) == [1]:
                    TP += 1
                elif list(a) == [0]:
                    FN += 1
            else:
                if list(a) == [1]:
                    TN += 1
                elif list(a) ==[0]:
                    FP += 1
            pro = clf.predict_proba(X)
            y_pre.append(pro[0][1])
            y_label.append(Y_din[new_key])
        save = pd.DataFrame(real_predict,columns=['result','day','fecture_day'])
    if state:        
        save.to_csv('data/'+name+'test_result.csv',header=True,index=False)
    try:
        Accuracy = (TP+FN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1score = 2*Precision*Recall/(Precision+Recall)
        print(TP,TN,FP,FN)
    except:
        Accuracy = 0
        F1score = 0
    return y_pre,y_label,predict,Accuracy,F1score,save  