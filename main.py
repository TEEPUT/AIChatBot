#=====================================
# Flask 웹서버 메인 프로그램
#=====================================
import socket
import pandas as pd
from flask import Flask, render_template, request
from myutil.mylib import Read_xlsx_Data

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('hello.html')
                       
@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        input_data = request.form["input_data"]
        print('***input_data : ', input_data)
        _file = './data/mydata.xlsx'
        _list = Read_xlsx_Data(_file)
        df = pd.DataFrame(_list[1:], columns=_list[0])
        result = viewPage(df)
        return result
    except Exception as ee:
        print('***error : ', ee)

def viewPage(df):
    # 역순으로 재정렬 (최신데이터를 맨 위로 올림)
    df = df.sort_index(ascending=False)
    sResult = "<!DOCTYPE html><html><head>"
    sResult += "<meta charset='utf-8'></head><body>"

    fields = df.columns.tolist()
    sResult += "<table border=1 role='table' bordercolor='green'>"
    sResult += "<thead role='rowgroup'>"
    sResult += "<tr role='row' bgcolor='#b9b922'>"
    for _f in fields:
        sResult += "<th role='columnheader'>" + _f + "</th>"

    sResult += "</tr></thead>"
    sResult += "<tbody role='rowgroup'>"
    for index, row in df.iterrows():
        sResult += "<tr role='row'>"
        for _f in fields:
            if str(row[_f]) == '':
                sResult += "<td role='cell'>-</td>"
            else:
                sResult += "<td role='cell'>" + str(row[_f]) + "</td>"

        sResult += "</tr>"
    sResult += "</tbody></table>"
    sResult += "</body></html>"

    return sResult

if __name__ == '__main__':
    _myip = socket.gethostbyname(socket.gethostname())
    app.run(host=_myip, port=9999, debug=True)

