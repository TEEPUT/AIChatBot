#=====================
# csv 파일을 읽어, 내용확인
#=====================
import json
import re
from konlpy.tag import Okt
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud

inputFileName = './etnews.kr_facebook_2016-01-01_2018-08-01_4차산업혁명(한빛출판네트워크)'
data = json.loads(open(inputFileName+'.json', 'r', encoding = 'utf8').read())
print('data = ', data)

message = ''
for item in data:
    if 'message' in item.keys():
        message = message + re.sub(r'[^\w]', ' ', item['message']) + ''
print('message = ', message)
print('='*40)

nlp = Okt()
message_N = nlp.nouns(message)
print('message_N = ', message_N)
print('='*40)

count = Counter(message_N)
print('count =', count)
print('='*40)

word_count = dict()
for tag, counts in count.most_common(80):
    if( len ( str ( tag ) ) > 2 ):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))
print('='*40)

font_path = "c:/Windows/fonts/malgun.ttf" # 맑은고딕체를 설정
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family = font_name)

plt.figure(num='텍스트빈도분석', figsize = (12, 7))
plt.title('4차 산업혁명 기사를 보고')
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)
sorted_Keys = sorted(word_count, key = word_count.get, reverse = True)
sorted_Values = sorted(word_count.values(), reverse = True)
plt.bar(range(len(word_count)), sorted_Values, align = 'center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation = 'vertical') #horizontal
plt.show()

#워드클라우드로 시각화
wc = WordCloud(font_path, background_color = 'ivory', width = 800, height = 600)
cloud = wc.generate_from_frequencies(word_count)
plt.figure(num='텍스트빈도분석', figsize = (6, 6))
plt.imshow(cloud)
plt.axis('off') # 아웃라인
plt.show()

# 이미지로 저장
cloud.to_file(inputFileName + '_cloud.jpg')