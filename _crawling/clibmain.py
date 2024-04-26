#----------------------------
# 크롬드라이버를 활용한 크롤링
#----------------------------
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import datetime
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

# 브라우저 생성
CoffeeBean_URL = "https://www.coffeebeankorea.com/store/store.asp"
#wd = webdriver.Chrome("C:\\_seok\\WebDriver\\chromedriver.exe")
wd = webdriver.Chrome(options=chrome_options)
wd.get(CoffeeBean_URL)

#[CODE 1]
def CoffeeBean_store(result):
    for i in range(1, 10):  # 매장 수만큼 반복
        wd.get(CoffeeBean_URL)
        time.sleep(1)  # 웹페이지 연결할 동안 2초 대기
        try:
            wd.execute_script("storePop2(%d)" % i)
            time.sleep(1)  # 스크립트 실행할 동안 2초 대기
            html = wd.page_source
            soupCB = BeautifulSoup(html, 'html.parser')
            store_name_h2 = soupCB.select("div.store_txt > h2")
            store_name = store_name_h2[0].string
            print(store_name)  # 매장 이름 출력하기
            
            store_info = soupCB.select("div.store_txt > table.store_table > tbody > tr > td")
            store_address_list = list(store_info[2])
            store_address = store_address_list[0]
            store_phone = store_info[3].string
            result.append([store_name] + [store_address] + [store_phone])
        except Exception as e:
            print("An error occurred: ", e)
            continue

#[CODE 0]
def main():
    result = []
    print('CoffeeBean store crawling >>>>>>>>>>>>>>>>>>>>>>>>')
    CoffeeBean_store(result)  # [CODE 1]
    CB_tbl = pd.DataFrame(result, columns=('store', 'address', 'phone'))
    CB_tbl.to_csv('./CoffeeBean.csv', encoding='cp949', mode='w', index=True)

if __name__ == '__main__':
    main()
