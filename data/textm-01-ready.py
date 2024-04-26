#-----------------------------------------------------------
def okt_tokenizer(text):
 tokens = okt.morphs(text)
 return tokens
#=====================
# python -m pip install sikit-learn
# 텍스트마이닝 - 감성분석
#=====================
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from konlpy.tag import Okt
import re

okt = Okt()

#warning 메시지 표시 안함
import warnings
warnings.filterwarnings(action = 'ignore')

#-----------------------------------------------------------
train_df = pd.read_csv('./ratings_train(한빛출판네트워크).txt',
    encoding = 'utf8', sep = '\t')

# 결측치 제거 : null 제거
train_df = train_df[train_df['document'].notnull()]

print(train_df['label'].value_counts())
print('01', '=' * 40)

# 한글이 아닌 문자 제거
train_df['document'] = train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

print(train_df.head())
print('02', '=' * 40)

#-----------------------------------------------------------
file_name = './코로나_naver_news(한빛출판네트워크).json'
with open(file_name, encoding = 'utf8') as _file:
    data = json.load(_file)

print(data)
print('03', '=' * 40)

data_title = []
data_description = []

for item in data:
 data_title.append(item['title'])
 data_description.append(item['description'])

print(data_title)
print('04', '=' * 40)

print(data_description)
print('05', '=' * 40)

data_df = pd.DataFrame({'title':data_title, 'description':data_description})

print(data_df)
print('06', '=' * 40)

#-----------------------------------------------------------
print('잠시만 기다리세요(약 5분)====================')

tfidf = TfidfVectorizer(tokenizer = okt_tokenizer,
    ngram_range = (1, 2), min_df = 3, max_df = 0.9)
tfidf.fit(train_df['document'])
train_tfidf = tfidf.transform(train_df['document'])

print('조금만 더 기다리세요(약 2분)==================')

SA_lr = LogisticRegression(random_state = 0)
SA_lr.fit(train_tfidf, train_df['label'])
LogisticRegression(random_state = 0)

params = {'C': [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid = params,
    cv = 3, scoring = 'accuracy', verbose = 1)

SA_lr_grid_cv.fit(train_tfidf, train_df['label'])
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))
print('07', '=' * 40)

#최적 매개변수의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_

#-----------------------------------------------------------
#1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
data_title_tfidf = tfidf.transform(data_df['title'])

#2) 최적 매개변수 학습 모델에 적용하여 감성 분석
data_title_predict = SA_lr_best.predict(data_title_tfidf)

#3) 감성 분석 결과값을 데이터프레임에 저장
data_df['title_label'] = data_title_predict

print(data_df['title_label'])
print('08', '=' * 40)

#-----------------------------------------------------------
#1) 분석할 데이터의 피처 벡터화 ---<< description >> 분석
data_description_tfidf = tfidf.transform(data_df['description'])

#2) 최적 매개변수 학습 모델에 적용하여 감성 분석
data_description_predict = SA_lr_best.predict(data_description_tfidf)

#3) 감성 분석 결과값을 데이터프레임에 저장
data_df['description_label'] = data_description_predict

print(data_df['description_label'])
print('09', '=' * 40)

#-----------------------------------------------------------
data_df.to_csv('./(정리)코로나_naver_news.csv', encoding = 'utf8')
print('10', '=== 파일이 저장 되었습니다.')
