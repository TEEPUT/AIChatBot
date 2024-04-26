#=====================
# 전처리-1 : 세미콜론(;) 으로 된 csv 파일을 읽어, xlsx 로 저장
#=====================
import pandas as pd

r_df = pd.read_csv('./winequality-red.csv', sep = ';', header = 0)
w_df = pd.read_csv('./winequality-white.csv', sep = ';', header = 0)

r_df.to_excel('./winequality-red2.xlsx', index=False)
w_df.to_excel('./winequality-white2.xlsx', index=False)
