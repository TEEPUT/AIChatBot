#================================
#AI API
#================================
import sys, io
import requests
import json
import datetime as dt
from mylib import mylib_Write_DB


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
ai_band = "myai"
ai_model = "Style_HC"
AI_URL = "http://192.168.50.35:9999/api/get_data"
SECRET_KEY = "AAAAAAAAAAAABBBCCC111"

#------------------------------------------------------
class ChatbotMessageSender:
    ep_path = AI_URL

    def req_message_send(self):
        self.ep_path = AI_URL
        request_body = {
            'key': SECRET_KEY,
            'question': myquery,
            'event': 'send'
        }

        custom_headers = {
            'Content-Type': 'application/json;UTF-8'
        }
        encode_request_body = json.dumps(request_body).encode('UTF-8')
        response = ''
        try:
            response = requests.post(headers=custom_headers, url=self.ep_path, data=encode_request_body)
        except Exception as ee:
            _msg = '(', self.ep_path, ') AI API 접속 주소를 확인하세요'
            response = requests.models.Response.text = str(_msg)
        return response

#--------------------------------------------------------
if __name__ == '__main__':
    myquery = ''
    quest1 = sys.argv[1]

    myai = ChatbotMessageSender()
    myquery = quest1
    res1 = myai.req_message_send()

    json_dict = json.loads(res1.text)

    sResult = json_dict['answer']

    itstime = dt.datetime.now()
    THISTIME = itstime.strftime("%Y%m%d%H:%M:%S:") + str(itstime.microsecond)[:3]
    
    try:
        mylib_Write_DB(THISTIME, ai_model, quest1, sResult)
    except Exception as ee:
        print('DB write error : ', ee, '\n')

    print(sResult)

