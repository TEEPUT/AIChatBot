#=====================
# csv 파일을 읽어, 내용확인
#=====================
import pandas as pd
import numpy as np

data = pd.read_csv('./보건복지부_공공보건 의료기관 현황_20161231.csv',
    index_col = 0, encoding = 'cp949')

print(data.head())
print('01', '=' * 40)

# 주소에서 시도, 군구 정보 분리
addr = pd.DataFrame ( data['주소'].apply(lambda v: v.split()[:2]).tolist(),
    columns = ('시도', '군구'))

print(data['주소'])
print('02', '=' * 40)

print(addr.head())
print('03', '=' * 40)

print(addr['시도'].unique())
print('04', '=' * 40)

print(addr[addr['시도'] == '창원시'])
print('05', '=' * 40)

addr.iloc[27] = ['경상남도', '창원시']
addr.iloc[31] = ['경상남도', '창원시']

print(addr.iloc[27], '\n\n', addr.iloc[31])
print('06', '=' * 40)

print(addr[addr['시도'] == '경산시'])
print('07', '=' * 40)

addr.iloc[47] = ['경상남도', '경산시']

print(addr[addr['시도'] == '천안시'])
print('08', '=' * 40)

addr.iloc[209] = ['충청남도', '천안시']
addr.iloc[210] = ['충청남도', '천안시']

print(addr['시도'].unique())
print('09', '=' * 40)

addr_aliases = {'경기':'경기도', '경남':'경상남도',
                '경북':'경상북도', '충북':'충청북도',
                '서울시':'서울특별시',
                '부산특별시':'부산광역시',
                '대전시':'대전광역시',
                '충남':'충청남도',
                '전남':'전라남도',
                '전북':'전라북도'}

addr['시도'] = addr['시도'].apply(lambda v: addr_aliases.get(v, v))

print(addr['시도'].unique())
print('10', '=' * 40)

print(addr['군구'].unique())
print('11', '=' * 40)

print(addr[addr['군구'] == '아란13길'])
print('12', '=' * 40)

addr.iloc[75] = ['제주특별자치도', '제주시']

addr['시도군구'] = addr.apply(lambda r: r['시도'] + ' ' + r['군구'], axis = 1)

print(addr.head())

print('13', '=' * 40)

addr['count'] = 0

print(addr.head())
print('14', '=' * 40)

addr_group = pd.DataFrame(addr.groupby(['시도', '군구', '시도군구'],
    as_index = False).count())
print(addr_group.head())
print('15', '=' * 40)

addr_group = addr_group.set_index("시도군구")
print(addr_group.head())
print('16', '=' * 40)

population = pd.read_excel('./(정리)행정구역_시군구_별__성별_인구수_20231124140845.xlsx')
                           
print(population.head())
print('17', '=' * 40)

population = population.rename(columns = {'행정구역(시군구준)별(1)': '시도', '행정구역(시군구준)별(2)': '군구'})
                                          
print(population.head())
print('18', '=' * 40)

# Warning 에러 안 보이도록 설정
pd.set_option('mode.chained_assignment', None)

for element in range(0,len(population)):
    population['군구'][element] = population['군구'][element].strip()

population['시도군구'] = population.apply(lambda r: r['시도'] + ' ' + r['군구'], axis = 1)
print(population.head())
print('19', '=' * 40) 

population = population[population.군구 != '소계']
population = population.set_index("시도군구")

print(population.head())
print('20', '=' * 40) 

addr_population_merge = pd.merge(addr_group,population,
    how = 'inner', left_index = True, right_index = True)

print(addr_population_merge.head())
print('21', '=' * 40) 

local_popu = addr_population_merge[['시도_x', '군구_x',
    'count', '총인구수 (명)']]

print(local_popu.head())
print('22', '=' * 40) 

#컬럼 이름 변경
local_popu = local_popu.rename(
    columns = {'시도_x':'시도', '군구_x': '군구',
                '총인구수 (명)': '인구수'})

popu_count = local_popu['count']
local_popu['ratio'] = popu_count.div(
    local_popu['인구수'], axis = 0) * 100000

print(local_popu.head())
print('23', '=' * 40) 

local_popu.to_excel('./blockmap_인구대비_공공보건의료기관비율.xlsx',
    index = '시도군구')

a = pd.read_excel('./blockmap_인구대비_공공보건의료기관비율.xlsx',
    index_col = 0)

print(a.head())
print('24', '=' * 40) 