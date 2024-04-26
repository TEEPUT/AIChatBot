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
