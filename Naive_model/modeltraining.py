from bigram import training,model_train
from pretreatment import market,spilcing
from transform import Onehot_encode
from Naive_train import Naive,predict_data
from backtest import judge

#主程序
def modeltrain(name,fee):
    bestacy = 0
    betterF1 = 0
    n = 1
    stage = 0
    marketdata = market('data/source/'+name+'data.csv',fee=fee)  #读取数据
    textdata = spilcing('data/trainingtext')  #读取文本
    model_train(textdata,'model/bigram/'+ name +'bi-gram.model')  #bi-gram模型训练
    textdata = training(textdata,'model/bigram/'+ name +'bi-gram.model')  #导入bi-gram模型
    
    #交叉验证选取最优参数
    for score in [0.1,0.2,0.3,0.4,0.5]:
        for real_mindf in [0.01,0.02,0.03,0.04,0.05]:
            Onehot_encode(textdata,'model/Onehot-train/'+name+'.pickle',round(score,2),round(real_mindf,3))  #独热编码的转换
            offset,right = Naive(textdata,marketdata,'model/bayes model/'+name+'.pickle',n)  #bayes模型训练
            y_pre,y_label,predict,Accuracy,F1score,save = predict_data(textdata,marketdata,'model/bayes model/'+name+'.pickle',   #将数据导入模型预测
                                                                    n, offset, stage, name,state=0)
            #选取最佳参数
            if Accuracy > bestacy and F1score<=0.9:
                bestacy = Accuracy
                betterF1 = F1score
                bestparameter = {'最佳阈值分数':score , '最佳词频':real_mindf}
    
    # 利用最佳参数重新训练模型并提取最佳数据
    Onehot_encode(textdata,'model/Onehot-train/'+name+'.pickle',round(bestparameter['最佳阈值分数'],2),round(bestparameter['最佳词频'],3))
    offset,right = Naive(textdata,marketdata,'model/bayes model/'+name+'.pickle',n)
    y_pre,y_label,predict,Accuracy,F1score,save = predict_data(textdata,marketdata,'model/bayes model/'+name+'.pickle', 
                                                            n, offset, stage, name, state=1)
    money,shape_rate = judge(save,'data/source/'+name+'data.csv',fee)
    print('最佳分数阈值为：{}；\n最佳词频为：{}'.format(bestparameter['最佳阈值分数'],bestparameter['最佳词频']))
    return Accuracy,F1score,shape_rate
        
if __name__ == '__main__':
    name = '动力煤'
    fee = 100
    Accuracy,F1score,shape_rate = modeltrain(name,fee)
    print('准确率为：{}；\nF1得分为：{}；\n对应夏普率为：{}；'.format(Accuracy,F1score,shape_rate))
