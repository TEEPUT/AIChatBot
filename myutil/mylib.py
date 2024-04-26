#===========================
# 공통 라이브러리
# 지정한 Excel 파일의 데이터 읽어 오기
#===========================
import pandas as pd
def mylib_Read_xlsx_Data(_file):
    _df = pd.read_excel(_file)
    _df = _df.fillna('')
    _fields = _df.columns.tolist()
    #print(_file)

    _array = []
    _array.append([])
    for _f in _fields:
        _array[0].append(_f)

    for _f in _fields:
        _index = 1
        for _a in _df[_f]:
            if _f in _fields[0]:
                _array.append([])
            _array[_index].append(_a)
            _index += 1

    return _array


#===========================
# 공통 라이브러리
# 데이터프레임을 웹에 보이기
#===========================
def mylib_ViewPage(df, input_data):
    # 역순으로 재정렬 (최신데이터를 맨 위로 올림)
    df = df.sort_index(ascending=False)

    sResult = "<!DOCTYPE html><html><head>"
    sResult += "<meta charset='utf-8'>"
    sResult += "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    sResult += "</head><body>"
    sResult += "<div><a href='/'>Home</a></div>"
    sResult += "<div>" + input_data + "</div>"
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

    return(sResult)

#===========================
# 공통 라이브러리
# MySQL DB 에 저장
#===========================
import pymysql
def mylib_Write_DB(_datetime, _model, _question, _answer):
    conn = pymysql.connect(host="localhost",
              port=int("3306"),
              user="aiya",
              password="10041004",
              db="chatbot",
              charset="utf8")
    sql = "INSERT INTO tbl_ai_comm (indate, model, question, answer) "
    sql += " VALUES (%s, %s, %s, %s)"

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (_datetime, _model, _question, _answer))
                conn.commit()
    except Exception as ee:
        print('DB write error sql=', sql, ' || ee=', ee)

#===========================
# 공통 라이브러리
# MySQL DB 에서 데이터 읽기
#===========================
import pymysql

def mylib_Read_DB(_sql):
    sResult = "<thead><tr><th>시간</th><th>모델</th>"
    sResult += "<th>질문</th><th>응답</th></tr></thead>"

    conn = pymysql.connect(host="localhost",
             port=int("3306"),
             user="aiya",
             password="10041004",
             db="chatbot",
             charset="utf8") 

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_sql)
                result = cur.fetchall()
                for data in result:
                    sResult += "<tr>"
                    sResult += "<td>" + str(data[0]) + "</td>"
                    sResult += "<td>" + str(data[1]) + "</td>"
                    sResult += "<td>" + str(data[2]) + "</td>"
                    sResult += "<td>" + str(data[3]) + "</td>"
                    sResult += "</tr>"

    except Exception as ee:
        print('DB read error sql=', _sql, ' || ee=', ee)
    return sResult
