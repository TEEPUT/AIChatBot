#=====================
# 결과시각화 : 부분 회귀 플롯
#=====================
import pandas as pd
from statsmodels.formula.api import ols, glm
import matplotlib.pyplot as plt
import statsmodels.api as sm

wine = pd.read_excel('./wine.xlsx')
wine.columns = wine.columns.str.replace(' ', '_')

# 선형회귀분석
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'
regression_result = ols(Rformula, data = wine).fit()

#--------------
# 부분 회귀 플롯 그리기
#--------------
#others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))
#p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)
#plt.show()

fig = plt.figure(figsize = (10, 10))
sm.graphics.plot_partregress_grid(regression_result, fig = fig)
plt.show()
