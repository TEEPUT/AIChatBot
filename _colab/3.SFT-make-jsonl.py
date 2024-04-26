#===========================
# Fine Tunning
#============================
import pandas as pd
import re
import json
#-------------------------------------------
# 전처리 함수 정의
#-------------------------------------------
def preprocess(text):
    if type(text) != str:
        text = '비어있는 데이터'

    # 한글, 영문, 숫자 및 공백을 제외한 모든 문자 제거
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)
    # 연속된 공백을 하나로 만들고 문자열 양 끝의 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()

    return text
#-------------------------------------------
# 메인 프로그램
#-------------------------------------------
def main(input_xlsx, output_json):
    data = pd.read_excel (input_xlsx)
    print('============================')
    print (data)

    print('============================')

    # 전처리
    data['prompt'] = data['prompt'].apply(preprocess)
    data['completion'] = data['completion'].apply(preprocess)

    # 데이터프레임에서 prompt와 completion의 token 수 계산
    try:
        data['prompt_tokens'] = data['prompt'].apply(lambda x: len(x.split()))
        data['completion_tokens'] = data['completion'].apply(lambda x: len(str(x).split()))
    except Exception as ee:
        print('exception ******* ', ee, 'data = ', data['completion'])
        data['completion_tokens'] = 0

    # prompt와 completion의 token 합계를 계산
    data['tokens'] = data['prompt_tokens'] + data['completion_tokens']

    # prompt, completion, token 합계의 내용을 리스트로 변환
    data_list = data.to_dict('records')

    # 필터링된 데이터를 리스트로 변환
    # 메모리 사이즈 때문에 전체 갯수 1000개로 제한
    filtered_data_list = [{'prompt': item['prompt'], 'completion': item['completion'],
        'tokens': item['tokens']} for item in data_list]
    
    # JSON 형식으로 저장
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(filtered_data_list, json_file, ensure_ascii=False, indent=4)
#-------------------------------------------
# 메인 함수 실행
if __name__ == '__main__':
    input_xlsx = './train_data/hair_styles.xlsx'
    output_json = './train_data/SFT_output.jsonl'

    main(input_xlsx, output_json)
