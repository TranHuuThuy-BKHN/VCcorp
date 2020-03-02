import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


class Gender:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        # vector hóa tf-idf, ở đây ta sẽ không dùng nó,
        # do lượng dữ liệu ~ 2tr, kích thước của từ điển cũng kha khá nên việc lưu trữ là rất khó khăn
        self.tf_idf = TfidfVectorizer(stop_words=None, tokenizer=None, encoding='utf-8', lowercase=True)
        self.tf_idf.fit(X)

        # vector hoa word2vec
        sentences = [str(w).lower().split() for w in self.X]
        self.word2vec = Word2Vec(sentences=sentences, window=1, size=50, sg=1, min_count=1)

        self.scale = {'mimax': MinMaxScaler(), 'stas': StandardScaler()}

        self.model = {'logistic':LogisticRegression(random_state=42), 'svm':LinearSVC(C=1, random_state=42)}

    # vector hóa dữ liệu text X và chuẩn hóa nó
    def transform(self, X, type='word2vec', scale_name='mimax'):
        res = []
        if type == 'tf-idf':
            res = self.tf_idf.transform(X)
        elif type == 'word2vec':
            sentences = [str(w).lower().split() for w in X]
            for i, sentence in enumerate(sentences):
                res.append(np.mean(self.word2vec.wv[sentence], axis=0))
        return self.scale[scale_name].fit_transform(res)

    def train(self, model_name='logistic', cv=5, scoring='roc_auc'):
        # đánh giá mô hình qua cross-validation
        y_cross = cross_val_score(self.model[model_name], self.X, self.y, cv=cv,
                                  scoring=scoring)
        print(model_name + ' cross-validation ' + scoring + ' ' + str(cv) + '-fold ', y_cross)
        print('Mean :', np.mean(y_cross))


path_male = 'Data/data_gender/data_male.tsv'
path_female = 'Data/data_gender/data_female.tsv'

data_male = pd.read_csv(path_male, sep='\t').to_numpy()[:, 0]
y_male = np.ones(np.shape(data_male)[0], dtype=int)

data_female = pd.read_csv(path_female, sep='\t').to_numpy()[:, 0]
y_female = -np.ones(np.shape(data_female)[0], dtype=int)

X = np.array([str(x).lower() for x in np.concatenate((data_male, data_female))])
y = np.concatenate((y_male, y_female))

gender = Gender(X, y)
# vector hóa dữ liệu
gender.X = gender.transform(X, type='word2vec')
gender.train(model_name='logistic')
gender.train(model_name='svm')
