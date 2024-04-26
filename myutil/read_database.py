#================================
# Read DataBase MySQL
#================================
import sys, io
import requests
from mylib import mylib_Read_DB

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

if __name__ == '__main__':

    sResult = "<html><head>"
    sResult += "<meta charset='utf-8'>"
    sResult += "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    sResult += "</head><body>"

    sql = "SELECT * FROM tbl_ai_comm "
    sql += " ORDER BY indate DESC "

    sResult += "<table border=1><tbody>"
    sResult += mylib_Read_DB(sql)
    sResult += "</tbody></table>"
    sResult += "</body></html>"
    
    print(sResult)
