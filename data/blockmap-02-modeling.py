#=====================
# xlsx 파일을 읽어, 내용확인
#=====================
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, style
from matplotlib import font_manager, rc

local_popu = pd.read_excel('./blockmap_인구대비_공공보건의료기관비율.xlsx',
index_col = 0)

print(local_popu.head())
print('01', '=' * 40) 

font_location = 'C:/Windows/Fonts/H2MKPB.ttf'
font_name = font_manager.FontProperties(
    fname = font_location).get_name()
rc('font', family = font_name)

_count = local_popu[['count']]
_count = _count.sort_values('count', ascending = False)

plt.rcParams["figure.figsize"] = (25, 5)
_count.plot(kind = 'bar', rot = 90)
plt.show()

_ratio = local_popu[['ratio']]
_ratio = _ratio.sort_values('ratio', ascending = False)

plt.rcParams["figure.figsize"] = (25, 5)
_ratio.plot(kind = 'bar', rot = 45)
plt.show()
