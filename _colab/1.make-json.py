#=====================
# 학습을 위한 json 데이터셋 생성
#=====================
import pandas as pd
import re
import json
#-------------------------------------------
# 전처리 함수 정의
def preprocess(text):
    if type(text) != str:
        text = '비어있는 데이터'

    # 한글, 영문, 숫자 및 공백을 제외한 모든 문자 제거
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)
    # 연속된 공백을 하나로 만들고 문자열 양 끝의 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()

    return text
#-------------------------------------------
# JSON 파일 작성
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
#-------------------------------------------
# 메인 함수 정의
def main(input_xlsx, output_json):
    df = pd.read_excel (input_xlsx)
    print('============================')
    print (df)
    print('============================')

    # 전처리
    df['prompt'] = df['prompt'].apply(preprocess)
    df['completion'] = df['completion'].apply(preprocess)
    _sentences = []
    for index, row in df.iterrows():
        p, c = row['prompt'], row['completion']
        _sen = []
        _sen.append((p, c)) # 쌍으로 묶어 준다
        _sentences.extend(_sen)

    # 결과 저장
    _data = [{'question': q, 'answer': a} for q, a in _sentences]
    write_json(_data, output_json)
#-------------------------------------------
# 메인 함수 실행
if __name__ == '__main__':
    input_xlsx = './train_data/hair_styles.xlsx'
    output_json = './train_data/output.json'

    main(input_xlsx, output_json)
