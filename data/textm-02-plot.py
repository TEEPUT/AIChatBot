#-----------------------------------------------------------
def okt_tokenizer(text):
 tokens = okt.morphs(text)
 return tokens
#-----------------------------------------------------------
#=====================
# python -m pip install sikit-learn
# 텍스트마이닝 - 감성분석 - 시각화
#=====================
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

okt = Okt()

#warning 메시지 표시 안함
import warnings
warnings.filterwarnings(action = 'ignore')

#-----------------------------------------------------------
data_df = pd.read_csv('./(정리)코로나_naver_news.csv',
    index_col=0, encoding = 'utf8')

print(data_df.head())
print('01', '=' * 40)

print(data_df['title_label'].value_counts())
print(data_df['description_label'].value_counts())
print('02', '=' * 40)

#-----------------------------------------------------------
columns_name = ['title', 'title_label', 'description', 'description_label']
NEG_data_df = pd.DataFrame(columns = columns_name)
POS_data_df = pd.DataFrame(columns = columns_name)

for i, data in data_df.iterrows():
    title = data["title"]
    description = data["description"]
    t_label = data["title_label"]
    d_label = data["description_label"]

    if d_label == 0: #부정 감성 샘플만 추출
        NEG_data_df = NEG_data_df._append(pd.DataFrame([[title, t_label, description,
            d_label]],columns = columns_name), ignore_index = True)
    else : #긍정 감성 샘플만 추출
        POS_data_df = POS_data_df._append(pd.DataFrame([[title, t_label, description,
            d_label]], columns = columns_name), ignore_index = True)
        
#-----------------------------------------------------------
print(NEG_data_df)
print('03', '=' * 40)

print(POS_data_df)
print('04', '=' * 40)

print(len(NEG_data_df), len(POS_data_df))
print('05', '=' * 40)

#-----------------------------------------------------------
POS_description = POS_data_df['description']
POS_description_noun_tk = []
POS_description_noun_join = []

for d in POS_description:
    POS_description_noun_tk.append(okt.nouns(d)) #명사 형태소만 추출

print(POS_description_noun_tk) #작업 확인용 출력
print('06', '=' * 40)

for d in POS_description_noun_tk:
    d2 = [w for w in d if len(w) > 1] #길이가 1보다 큰 토큰만 추출
    POS_description_noun_join.append(" ".join(d2)) #토큰 연결하여 리스트 구성

print(POS_description_noun_join) #작업 확인용 출력
print('07', '=' * 40)

#-----------------------------------------------------------
NEG_description = NEG_data_df['description']
NEG_description_noun_tk = []
NEG_description_noun_join = []

for d in NEG_description:
    NEG_description_noun_tk.append(okt.nouns(d)) #명사 형태소만 추출

print(NEG_description_noun_tk) #작업 확인용 출력
print('08', '=' * 40)

for d in NEG_description_noun_tk:
    d2 = [w for w in d if len(w) > 1] #길이가 1보다 큰 토큰만 추출
    NEG_description_noun_join.append(" ".join(d2)) # 토큰 연결하여 리스트 구성

print(NEG_description_noun_join) #작업 확인용 출력
print('09', '=' * 40)

#-----------------------------------------------------------
POS_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df = 2)
POS_dtm = POS_tfidf.fit_transform(POS_description_noun_join)

POS_vocab = dict()

for idx, word in enumerate(POS_tfidf.get_feature_names_out()):
    POS_vocab[word] = POS_dtm.getcol(idx).sum()

POS_words = sorted(POS_vocab.items(), key = lambda x: x[1], reverse = True)

print(POS_words) #작업 확인용 출력
print('10', '=' * 40)

#-----------------------------------------------------------
NEG_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df = 2 )
NEG_dtm = NEG_tfidf.fit_transform(NEG_description_noun_join)

NEG_vocab = dict()

for idx, word in enumerate(NEG_tfidf.get_feature_names_out()):
    NEG_vocab[word] = NEG_dtm.getcol(idx).sum()

NEG_words = sorted( NEG_vocab.items(), key = lambda x: x[1], reverse = True)

print(NEG_words) #작업 확인용 출력
print('11', '=' * 40)

#-----------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm._get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname = font_location).get_name()
matplotlib.rc('font', family = font_name)

max = 15 #바 차트에 나타낼 단어의 수

plt.bar(range(max), [i[1] for i in POS_words[:max]], color = "blue")
plt.title("긍정 뉴스의 단어 상위 %d개" %max, fontsize = 15)
plt.xlabel("단어", fontsize = 12)

plt.ylabel("합", fontsize = 12)
plt.xticks(range(max), [i[0] for i in POS_words[:max]], rotation = 70)

plt.show()

plt.bar(range(max), [i[1] for i in NEG_words[:max]], color = "red")
plt.title("부정 뉴스의 단어 상위 %d개" %max, fontsize = 15)
plt.xlabel("단어", fontsize = 12)
plt.ylabel("합", fontsize = 12)
plt.xticks(range(max), [i[0] for i in NEG_words[:max]], rotation = 70)

plt.show()