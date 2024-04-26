#=====================
# 회귀분석- 자동차연비 예측
#=====================
import numpy as np
import pandas as pd

df = pd.read_excel('./auto-mpg.xlsx', index_col = 0)

print('데이터셋 크기: ', df.shape)
print(df.head())
print('01', '=' * 40)

# 사용하지 않는 컬럼 삭제
df = df.drop(['carname', 'origin', 'hp'], axis = 1, inplace = False)
print(df.head())
print('02', '=' * 40)

# 데이터셋의 정보 확인
print(df.info())
print('03', '=' * 40)

# 저장
df.to_excel('./auto-mpg-result.xlsx')