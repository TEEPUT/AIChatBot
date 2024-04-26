#=====================
# python –m pip install scikit-learn
# 회귀분석- 자동차연비 예측
# 사이킷런 패키지 활용
#=====================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel('./auto-mpg-result.xlsx')

# X, Y 분할하기
Y = df['mpg']

X = df.drop(['mpg'], axis = 1, inplace = False)

# 훈련용 데이터와 평가용 데이터 분할하기
# 데이터를 7:3 비율로 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# 선형 회귀 분석 : 모델 생성
reg = LinearRegression()

# 선형 회귀 분석 : 모델 훈련
reg.fit(X_train, Y_train)

# 질문과 답
print("연비를 예측하고 싶은 차의 정보를 입력해주세요.")
cylinders_1 = int(input("cylinders(8) : "))
displacement_1 = int(input("displacement(350) : "))
weight_1 = int(input("weight(3200) : "))
acceleration_1 = int(input("acceleration(22) : "))
model_year_1 = int(input("model_year(99) : "))
mpg_predict = reg.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1]])
print("이 자동차의 예상 연비(MPG)는 %.2f입니다." %mpg_predict)
