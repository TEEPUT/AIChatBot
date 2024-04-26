#=====================
# 전처리-2 : 파일 병합
#=====================
import pandas as pd

r_df = pd.read_excel('./winequality-red2.xlsx')
w_df = pd.read_excel('./winequality-white2.xlsx')

r_df.insert(0, column = 'type', value = 'red')
w_df.insert(0, column = 'type', value = 'white')

wine = pd.concat([r_df, w_df])
wine.to_excel('./wine.xlsx', index=False)