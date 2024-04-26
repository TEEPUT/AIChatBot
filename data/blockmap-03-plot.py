#=====================
# csv 파일을 읽어, 내용확인
#=====================
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.family': 'malgun gothic', 'font.size': 12})

draw_korea = pd.read_csv('./data_draw_korea(한빛출판네트워크).csv',
    index_col = 0, encoding = 'UTF-8')

print(draw_korea.head())
print('01', '=' * 40)

local_popu = pd.read_excel('./blockmap_인구대비_공공보건의료기관비율.xlsx',
    index_col = 0)

print(local_popu.head())
print('02', '=' * 40) 

draw_korea['시도군구'] = draw_korea.apply(
    lambda r: r['광역시도'] + ' ' + r['행정구역'], axis = 1)

draw_korea = draw_korea.set_index("시도군구")

print(draw_korea.head())
print('03', '=' * 40) 

draw_all = pd.merge(draw_korea, local_popu,
    how = 'outer', left_index = True, right_index = True)

print(draw_all.head())
print('04', '=' * 40) 

BORDER_LINES = [
    [(3, 2), (5, 2), (5, 3), (9, 3), (9, 1)], # 인천
    [(2, 5), (3, 5), (3, 4), (8, 4), (8, 7), (7, 7), (7, 9), (4, 9),(4, 7), (1, 7)], # 서울
    [(1, 6), (1, 9), (3, 9), (3, 10), (8, 10), (8, 9),(9, 9), (9, 8), (10, 8), (10, 5), (9, 5), (9, 3)], # 경기도
    [(9, 12), (9, 10), (8, 10)], # 강원도
    [(10, 5), (11, 5), (11, 4), (12, 4), (12, 5), (13, 5), (13, 4), (14, 4), (14, 2)], #충청남도
    [(11, 5), (12, 5), (12, 6), (15, 6), (15, 7), (13, 7), (13, 8), (11, 8), (11, 9), (10, 9), (10, 8)], # 충청북도
    [(14, 4), (15, 4), (15, 6)], # 대전시
    [(14, 7), (14, 9), (13, 9), (13, 11), (13, 13)], # 경상북도
    [(14, 8), (16, 8), (16, 10), (15, 10), (15, 11), (14, 11), (14, 12), (13, 12)], # 대구시
    [(15, 11), (16, 11), (16, 13)], # 울산시
    [(17, 1), (17, 3), (18, 3), (18, 6), (15, 6)], # 전라북도
    [(19, 2), (19, 4), (21, 4), (21, 3), (22, 3), (22, 2), (19, 2)], #광주시
    [(18, 5), (20, 5), (20, 6)], # 전라남도
    [(16, 9), (18, 9), (18, 8), (19, 8), (19, 9), (20, 9), (20, 10)], #부산시
]

def draw_blockMap(blockedMap, targetData, title, color):
    whitelabelmin = (
        max(blockedMap[targetData]) - min(blockedMap[targetData])
        ) * 0.25 + min(blockedMap[targetData])
    
    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot(index = 'y', columns = 'x', values = targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)

    plt.figure(figsize = (8, 13))
    plt.title(title)
    plt.pcolor(masked_mapdata, vmin = vmin, vmax = vmax,
        cmap = color, edgecolor = '#ffcc99', linewidth = 0.5)
    
    #지역 이름 표시
    for idx, row in blockedMap.iterrows():
        annocolor = 'black' #if row[targetData] > whitelabelmin else 'black'

        #광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시
        if row['광역시도'].endswith('시') and not row['광역시도'].startswith('세종'):
            dispname = '{}\n{}'.format(row['광역시도'][:2], row['행정구역'][:-1])

            if len(row['행정구역']) <= 2:
                dispname += row['행정구역'][-1]
            else:
                dispname = row['행정구역'][:-1]

            #서대문구, 서귀포시 같이 이름이 3자 이상이면 작은 글자로 표시
            if len(dispname.splitlines()[-1]) >= 3:
                fontsize, linespacing = 9.5, 1.5
            else:
                fontsize, linespacing = 11, 1.2

            plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight = 'bold',
                fontsize = fontsize, ha = 'center', va = 'center', color = annocolor,
                linespacing = linespacing)

    #시도 경계를 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c = 'black', lw = 4)

    plt.gca().invert_yaxis()
    plt.axis('off')

    cb = plt.colorbar(shrink = 1, aspect = 10)
    cb.set_label(datalabel)
    plt.tight_layout()
    plt.savefig('./blockmap_' + targetData + '.png')
    plt.show()

#-----------------------------------
# Main 실행
#-----------------------------------
draw_blockMap(draw_all, 'count', '행정구역별-공공보건의료기관수', 'brg')
draw_blockMap(draw_all, 'ratio', '행정구역별 인구수 대비 공공보건의료기관 비율', 'winter_r')
