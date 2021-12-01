from gensim.models import Phrases
import jieba
def model_train(data,model_path):
    phrases_model = Phrases(data, min_count=20, threshold=20)  
    phrases_model.save(model_path)

def training(din,model_path):
    new_din ={}
    a = []
    bigram = Phrases.load(model_path)  # 导入训练好的bi-gram模型
    print('正在训练')
    for key in din.keys():
        print('正在训练{}的语料'.format(key),end='\r')
        new_din[key] = []
        for news_amo in din[key]:
            for real_news in range(len(news_amo)):
                catch = bigram[news_amo[real_news]]  # 将字典中的词导入模型中进行处理
                for i in range(len(catch)):
                    if '_' in catch[i]:
                        catch[i] = catch[i].replace('_','')
                        if not catch[i] in a:
                            a.append(catch[i])
                        jieba.add_word(catch[i])
                new_din[key].append(catch) 
    with open('words.txt','w',encoding='utf-8-sig') as f:
        for n in a:
            f.write(n+'\n')  
    print('训练完成')
    return new_din   
