# encoding = utf-8
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#拼接各句子
def splicing(data):
    sentence =[]
    print('正在拼接', end='\r')
    for words in data:
        sentence.append(' '.join(str(i)for i in words))
    return sentence 

def train(train_list,model_name,rea_mindf):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95 , min_df=round(rea_mindf,4))  
    tfidf_X = tfidf_vectorizer.fit(train_list)
    word = tfidf_vectorizer.get_feature_names()
    with open(model_name,'wb') as f: 
        pickle.dump(tfidf_X,f) 
    return word

#提取特征词并转换为独热编码
def Onehot_encode(train_din, model_name, score, real_mindf):  
    train_list = []
    for key in train_din.keys():
        train_din[key] = splicing(train_din[key])
        train_list.extend(train_din[key])
        train_din[key] = [' '.join(str(i)for i in train_din[key])]
        print('正在拼接{}日的文本'.format(key),end='\r')
    print('拼接完成')
    word = train(train_list,model_name,real_mindf)
    with open(model_name,'rb') as k:
        tfidf_X = pickle.load(k)
    for new_key in train_din.keys():
        train_din[new_key] = tfidf_X.transform(train_din[new_key]).toarray()
        for i in range(len(train_din[new_key])):
            for j in range(len(train_din[new_key][i])):
                if train_din[new_key][i,j] >= score:
                    train_din[new_key][i,j] = 1
                else:
                    train_din[new_key][i,j] = 0 

