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
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('auto-mpg-result.xlsx')

print(df)
print('01', '=' * 40)

# X, Y 분할하기
Y = df['mpg']
X = df.drop(['mpg'], axis = 1, inplace = False)

print(Y)
print('02', '=' * 40)

print(X)
print('03', '=' * 40)

# 훈련용 데이터와 평가용 데이터 분할하기
# 데이터를 7:3 비율로 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

print('X_train = ', X_train)
print('04', '=' * 40)

# 선형 회귀 분석 : 모델 생성
reg = LinearRegression()

# 선형 회귀 분석 : 모델 훈련
reg.fit(X_train, Y_train)

# 선형 회귀 분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = reg.predict(X_test)

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

print('Y 절편 값: ', np.round(reg.intercept_, 2))
print('회귀 계수 값: ', np.round(reg.coef_, 2))

coef = pd.Series(data = np.round(reg.coef_, 2), index = X.columns)
print(coef.sort_values(ascending = False))
print('05', '=' * 40)

fig, axs = plt.subplots(figsize = (16, 16), ncols = 3, nrows = 2)
x_features = ['modely', 'accel', 'displ', 'weight', 'cylinder']
plot_color = ['r', 'b', 'y', 'g', 'r']
for i, feature in enumerate(x_features):
    row = int(i/3)
    col = i%3
    sns.regplot(x = feature, y = 'mpg', data = df, ax = axs[row][col], color = plot_color[i])
    
plt.show()
