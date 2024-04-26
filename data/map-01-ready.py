#=====================
# csv 파일을 읽어, 내용확인
#=====================
import pandas as pd
CB = pd.read_csv('./인천재능대학_지리정보분석_실습_E.csv',
    encoding='cp949', index_col = 0, header = 0)
print(CB.head())
print('=' * 40)

addr = []
for address in CB.address:
    addr.append(str(address).split())
print(addr)
print('=' * 40)

addr2 = []
for i in range(len(addr)):
    if addr[i][0] == "인천": addr[i][0] = "인천광역시"
    elif addr[i][0] == "인천시": addr[i][0] = "인천광역시"
    addr2.append(' '.join(addr[i]))
print(addr2)
print('=' * 40)

addr2 = pd.DataFrame(addr2, columns = ['address2'])
CB2 = pd.concat([CB, addr2], axis = 1 )
print(CB2.head())
print('=' * 40)

CB2.to_csv('./(정리)인천재능대학_지리정보분석_실습_E.csv',
    encoding = 'CP949', index = False)