#=====================
# 데이터탐색 : 종속변수 값 확인
#=====================
import pandas as pd

wine = pd.read_excel('./wine.xlsx')

print(wine.info())
print('=' * 40)

wine.columns = wine.columns.str.replace(' ', '_')
print(wine.head())
print('=' * 40)

print(wine.describe())
print('=' * 40)

print(sorted(wine.quality.unique()))
print('=' * 40)

wine.quality.value_counts()
print('=' * 40)