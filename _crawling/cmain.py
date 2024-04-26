#=======================
# main program
#=======================
import json
import pandas as pd
from CODE2 import getNaverSearch
from CODE3 import getPostData

def main():
    MAX_CNT = 1000  # 검색해 올 총 개수
    DISPLAY_CNT = 10  # 한번에 표시할 개수
    MAX_START_POINT = 1000  # MAX 검색시작위치
    node = 'news'  # 크롤링할 대상
    srcText = input('검색어를 입력하세요: ')
    # srcText = '중국음식'

    cnt = 0
    jsonResult = []
    
    jsonResponse = getNaverSearch(node, srcText, 1, DISPLAY_CNT)  # [CODE 2]
    total = jsonResponse['total']
    
    while ((jsonResponse != None) and (jsonResponse['display'] != 0)):
        for post in jsonResponse['items']:
            cnt += 1
            getPostData(post, jsonResult, cnt)  # [CODE 3]
        start = jsonResponse['start'] + jsonResponse['display']
        
        if start > MAX_START_POINT:
            start = 1
        if start > MAX_CNT:
            break
        jsonResponse = getNaverSearch(node, srcText, start, DISPLAY_CNT)

    print('전체 검색 : %d 건' % total)
    file_name = ("naver_%s_%s(%05d)" % (node, srcText, cnt))
    
    with open(file_name + '.json', 'w', encoding='utf8') as outfile:
        jsonFile = json.dumps(jsonResult, indent=4, ensure_ascii=False)
        outfile.write(jsonFile)

    out_df = pd.DataFrame(jsonResult)
    out_df.to_excel(file_name + '.xlsx', index=False)
    print("가져온 데이터 : %d 건" % cnt)
    print('%s SAVED' % file_name)

if __name__ == '__main__':
    main()
