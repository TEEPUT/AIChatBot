#=====================
# 데이터모델링 :
#=====================
import pandas as pd

wine = pd.read_excel('./wine.xlsx')
wine.columns = wine.columns.str.replace(' ', '_')

print(wine.groupby('type')['quality'].describe())
print('=' * 40)

print(wine.groupby('type')['quality'].agg(['mean', 'std']))
print('=' * 40)

#---------------
# t-검정
#---------------
from scipy import stats
from statsmodels.formula.api import ols, glm
r_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
w_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

print(stats.ttest_ind(r_wine_quality, w_wine_quality, equal_var = False))
print('=' * 40)

#---------------
# 선형회귀분석
#---------------
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'

regression_result = ols(Rformula, data = wine).fit()
print(regression_result.summary())
print('=' * 40)

#---------------
# 품질등급 예측
#---------------
sample1 = wine[wine.columns.difference(['quality', 'type'])]
sample1 = sample1[0:5][:]
print(sample1)
print('=' * 40)

sample1_predict = regression_result.predict(sample1)
print(sample1_predict)
print('=' * 40)

print(wine[0:5]['quality'])
print('=' * 40)

#=====================
# 결과시각화 : 히스토그램
#=====================
import pandas as pd
from statsmodels.formula.api import ols, glm
import matplotlib.pyplot as plt
import seaborn as sns

wine = pd.read_excel('./wine.xlsx')
wine.columns = wine.columns.str.replace(' ', '_')
r_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
w_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

#--------------
# 히스토그램 그리기
#--------------
sns.set_style('dark')
sns.distplot(r_wine_quality, kde = True, color = "red", label = 'red wine')
sns.distplot(w_wine_quality, kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()
plt.show()