#=======================
# kogpt 학습
#=======================
import os
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast

# GPU가 사용 가능한 경우 사용하고, 그렇지 않으면 CPU를 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = ', device)

# epoch 크기 조정 : 학습량
EPOCH = 1 # 1 ~ 1000 값
# 읽어 올 데이터셋
DATA_SET_JSON = './train_data/output.json'
# 학습한 모델 저장하는 폴더명
SAVED_MODEL_NAME = './kogpt-trained'

# KoGPT-2 모델 및 토크나이저 로드
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

# 모델을 디바이스(GPU가 사용 가능한 경우 GPU)로 이동
model.to(device)

# output.jason 파일에서 학습 데이터 로드
df = pd.read_json(DATA_SET_JSON, encoding='utf-8')

# 데이터를 학습 및 검증 세트로 분할
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def preprocess(df):
    input_ids = []
    attention_masks = []
    for index, row in df.iterrows():
        # 질문과 응답을 합쳐 하나의 텍스트로 생성
        text = row['question'] + tokenizer.eos_token + row['answer']
        # 텍스트를 토큰화하고 텐서로 변환
        tokens = tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=512,
            padding='max_length',
            truncation=True
        )
        
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

# 전처리된 데이터 생성
train_input_ids, train_attention_masks = preprocess(train_df)
val_input_ids, val_attention_masks = preprocess(val_df)

# DataLoader 생성
train_dataset = TensorDataset(train_input_ids, train_attention_masks)
train_dataloader = DataLoader(train_dataset, batch_size=8)

val_dataset = TensorDataset(val_input_ids, val_attention_masks)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# 모델 학습
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5, # 학습률 : 각 파라미터 업데이트의 크기
    betas=(0.9, 0.999), # 0.9 : 과거의 기울기 반영비율, 0.999 : 과거의 기울기 제곱의 평균 반영비율
    eps=1e-8, # 0으로 나누는 것 방지 위에 분모에 추가되는 작은 상수
    weight_decay=0, # L2 정규화를 적용하는 데 사용되는 가중치 감소계수, 0이 아니면 모델의 복잡성 감소, 과적합 방지
    amsgrad=False # Adam 알고리즘의 문제점을 개선한 amsgrad 모델 반영여부
)

for epoch in range(EPOCH):
    for i, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_masks = batch
        
        # 텐서를 디바이스(GPU가 사용 가능한 경우 GPU)로 이동
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        
        # 경사 초기화 및 순방향 전파
        optimizer.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_input_ids)
        
        # 손실 계산 및 역방향 전파
        loss = outputs.loss
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()
        
        # 손실 출력
        if i % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}')
# 모델 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i, batch in enumerate(val_dataloader):
        batch_input_ids, batch_attention_masks = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        
        # 모델을 사용하여 예측값 생성
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        
        # 예측값 중 가장 높은 확률의 토큰 인덱스를 선택
        _, predicted = torch.max(outputs.logits, 2)
        
        # 전체 예측 개수 및 정확한 예측 개수를 계산
        total += batch_input_ids.size(0)
        correct += (predicted == batch_input_ids).sum().item()
        
# 정확도 계산
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')
# 학습된 모델 저장
model.save_pretrained(SAVED_MODEL_NAME)