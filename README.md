<h1>📌 Intro</h1><br>
정석일 교수님의 강의를 들으면서 제작한 나만의 챗봇비서입니다 <br><br>

ChatGPT, Copilot 등 오픈 AI들은 어느 한 분야에 특화된 AI가 모든 분야에 대해 지식이 있어 내가 원하는 답변이 안나올 때도 있는데 그 점을 보안하고자 하여 한 분야만 딥러닝 시켜 필요한 답변만 주는 AI 모델을 만들었습니다 <br><br>

물론 AI 모델이 학습량과 학습시킬 데이터의 정확성 등 여러 요소들이 모여 높은 수준으로 완성되므로 개인이 할 수 있는 딥러닝에서는 한계가 존재합니다 <br><br>

제가 만든 상황에 맞춰서 헤어스타일링을 추천하는 AI는 약 500개의 데이터로 10000번 이상 학습시켰습니다 <br><br>

Python 라이브러리인 Transformers를 사용하여 학습시켰고 Colab의 GPU를 사용하여 학습시켰습니다 <br><br>

myutil 디렉토리 call_ai_api.py 파일의 AI_URL 부분의 자신의 로컬 ip로 입력하셔야 합니다 <br>
마찬가지로 main2.py 파일의 라우팅 호스트 ip를 자신의 로컬 ip로 입력하셔야합니다

call_ai_api.py:14 ```AI_URL = "http://[localhostip]:9999/api/get_data"```<br>
main2.py:147 ```app.run(host='[localhostip]', port=9999, debug=False)```

데이터셋은 인터넷 검색 및 크롤링 AI모델 사용 하여 구축하였습니다

<h1>📌 Implementation</h1><br>

![image](https://github.com/TEEPUT/AIChatBot/assets/129711481/79d1a052-4e38-4916-9abc-53edced17fb2)

npm start 로 nodejs 프론트엔드를 실행시키고<br>
python main2.py로 백엔드를 실행시킵니다<br>
백엔드와 프론트엔드를 실행시키고 http://localhost:5555/ 에서 동작하는 서비스를 볼 수 있습니다 <br>

<h3>필요 nodejs 라이브러리</h3>

>npm install express --save <br>
>npm install express-generator -g --save <br>
>npm install fs --save <br>
>npm install mysql --save <br>
>npm install path --save <br>
>npm install body-parser --save <br>
>npm i dotenv <br>
>npm i cors <br>
>npm i nodemon <br>
>npm install mobile-detect <br>
>npm install request --save <br>
>npm install jquery <br>
>npm install request-ip --save <br>
>npm install jsonwebtoken --save <br>
>npm install cookie-parser --save <br>
>npm install moment --save <br>
>npm install winston --save <br>
>npm install winston-daily-rotate-file --save <br>
>npm install ip --save <br>

<br>

<h3>필요 python 라이브러리</h3>

>pip install flask <br>
>python -m pip install selenium 크롤링 라이브러리<br>
>python -m pip install torch torchvision torchaudio 파이토치 딥러닝 라이브러리<br>
>pip install transformers <br>

<br>

![image](https://github.com/TEEPUT/AIChatBot/assets/129711481/fd9a9b98-bbe2-4a9e-a6f7-6d74f8b58a2a)
초기 화면입니다 채팅 형식으로 자신이 원하는 prompt 를 입력하시면 됩니다 ex) "오피스 헤어 추천해줘"<br><br>

답변은 로컬 환경(사양)에 따라 시간차가 있습니다<br>

![image](https://github.com/TEEPUT/AIChatBot/assets/129711481/05d7b9d0-9285-491f-9f63-34063d34c632)
디렉토리 구성

<h3>모델 학습 (_colab)</h3>
학습 데이터셋 생성: Excel 파일에서 데이터를 읽고, 전처리한 후 JSON 파일로 변환합니다. 이 데이터셋은 KoGPT 모델을 학습하기 위해 사용됩니다. <br><br>
KoGPT 모델 학습: 사전 학습된 KoGPT 모델을 로드하고, 학습 데이터셋을 사용하여 모델을 학습시킵니다. 이후 모델을 평가하고 저장합니다.<br><br>

학습 데이터셋 생성:
```python
def preprocess(text):
    # 전처리 함수 정의

def write_json(data, file_path):
    # JSON 파일 작성

def main(input_xlsx, output_json):
    # 메인 함수 정의

```
KoGPT 모델 학습:
```python
# GPU가 사용 가능한 경우 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epoch 크기 조정 : 학습량
EPOCH = 1 # 1 ~ 1000 값

# 전처리된 데이터 생성
train_input_ids, train_attention_masks = preprocess(train_df)
val_input_ids, val_attention_masks = preprocess(val_df)

# 모델 학습
model.train()

# 모델 평가
model.eval()
```
Fine Tuning:
```python
# 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

# 사전 학습된 언어모델(skt/kogpt2-base-v2) 로드
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# SFT_dataset 클래스를 사용하여 훈련 데이터셋 객체를 생성
train_dataset = SFT_dataset(data_path=args.data_path, tokenizer=tokenizer)

# 모델 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 모델 인퍼런스
generator = pipeline('text-generation', model=sOutput_FineTunning_Result_Directory, tokenizer=tokenizer)

```
이 코드는 데이터 전처리, 모델 설정, 학습 및 평가, 그리고 생성 작업을 수행하는 핵심 부분입니다<br><br>

