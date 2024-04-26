#=====================================
# Flask 웹서버 메인 프로그램
#=====================================
import sys, os
import socket
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from myutil.mylib import mylib_Read_xlsx_Data, mylib_ViewPage


#---------------------------------
# KoGPT
#---------------------------------
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

#--여기서부터 추가-Part-1------------
from copy import deepcopy
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForNextSentencePrediction
from KoChatGPT.colossalai_ChatGPT_230319.chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt_api.chatgpt_api_service import query_chatgpt
#--여기까지 추가-Part-1------------------------

#--여기서부터 추가-Part-2------------------------

# Set device based on CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('****** my device = ', device)

PRETRAINED_MODEL = "skt/kogpt2-base-v2"

MY_AI_MODEL = './kogpt-SFT-trained'
LORA_RANK = 0
actor = GPTActor(pretrained=MY_AI_MODEL, lora_rank=LORA_RANK).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, padding_side="right", model_max_length=5000)
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "</s>",
    "unk_token": "</s>",
})
tokenizer.pad_token = tokenizer.eos_token
initial_model = deepcopy(actor)

def request_AI(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = actor.generate(input_ids,
        max_length=512,
        do_sample=True,
        top_k=50,
        top_p=0.3,
        temperature=0.3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
    if output.find(input_text) == 0:
        output = output[len(input_text):]
        if output == '':
            output = "아직 학습을 하지 못 했습니다. 다른 질문을 해주세요~"
    return output
#--여기까지 추가-Part-2------------

''' 이 부분은 막음

# Define the pretrained model and tokenizer
PRETRAINED_MODEL = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    PRETRAINED_MODEL,
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>'
)

# Load the model
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL)
model.to(device) # Move the model to the appropriate device

# Define the request function
def request_AI(_req):
    # Generate a response using the model
    _res = model.generate(
        _req,
        max_length=128,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True
    )
    return _res
#---------------------------------
이 부분은 막음
'''
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('hello2.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        input_data = request.form["input_data"]
        print('***input_data : ', input_data)
        _file = './data/mydata.xlsx'
        _list = mylib_Read_xlsx_Data(_file)
        df = pd.DataFrame(_list[1:], columns=_list[0])
        result = mylib_ViewPage(df, input_data)
        return result

    except Exception as ee:
        print('***error : ', ee)

@app.route('/api/get_data', methods=['POST'])
def api_page():
    
    question = request.json['question']
    print('***question : ', question)
    apikey = request.json['key']

    if apikey != 'AAAAAAAAAAAABBBCCC111':
        return jsonify({'answer': 'not supported'})

    answer = ""

    try:
        #input_ids = tokenizer.encode(question, return_tensors='pt’)
        answer = request_AI(question)
        #answer = tokenizer.decode(generated[0])

        #answer = question + " : 학습이 된 후에 답변 드리겠습니다~"
    except Exception as ee:
        answer = "오류가 발생했습니다~" + ee

    print('***answer : ', answer)

    sys.stdout.flush()
    return jsonify({'answer': answer})


if __name__ == '__main__':
    #_myip = socket.gethostbyname(socket.gethostname())
    app.run(host='192.168.50.35', port=9999, debug=False)
