import openai
import sys
import datetime as dt
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

openai.api_key = "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"

model = "gpt-3.5-turbo"
ai_band = "OpenAI"
ai_model = "gpt-3.5-turbo"

def query_chatgpt(query):
    answer = ""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature=0.9
        )
        answer = response['choices'][0]['message']['content']
    except Exception as ee:
        print('Open AI error : ', ee)
        answer = "API_ERROR"

    return (answer)

def query_chatgpt_chatcompletion(_model, _query) :
    answer = 'None'

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _query}
    ]
    try:
        response = openai.ChatCompletion.create(
            model = _model,
            messages = messages
        )
        answer = response['choices'][0]['message']['content']
    except Exception as ee:
        print('chatGPT API error : ', ee)
        answer = "API_ERROR"

    return (answer)

def query_chatgpt_completion(_model, _query) :
    answer = 'None'
    try:
        response = openai.Completion.create(
            model = _model,
            prompt = _query
        )
        answer = response['choices'][0]['text']
    except Exception as ee:
        print('chatGPT API error : ', ee)
        answer = "API_ERROR"
                
    return (answer)

if __name__ == '__main__':
   print('사용법 : python chatgpt_api_service "오늘의 날씨는?"')
   ques1 = sys.argv[1]
   sResult = query_chatgpt(ques1)

   print('질의 : ', ques1)
   print('답변 : ', sResult)

