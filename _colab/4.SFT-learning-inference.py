#===========================
# Fine Tunning
# !pip install transformers==4.28.0
# !pip install --upgrade accelerate
#============================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import Dataset
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, pipeline
from transformers import Trainer, TrainingArguments
from transformers import AutoModelWithLMHead
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import argparse
import logging
import json
import copy
from copy import deepcopy
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    ),
}
IGNORE_INDEX = -100

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in list(state_dict.items())}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 

class SFT_dataset(Dataset):
 
    # 초기화 : 데이터셋 파일을 읽고, 데이터 전처리 후 토큰화 하여 저장

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")

        ## format
        pattern_instruction = 'prompt'
        pattern_input = 'input'
        pattern_output = 'completion'

        ## load dataset(데이터셋 내에 input은 없음)
        with open(data_path, "r", encoding='utf-8-sig') as json_file:
            list_data_dict = json.load(json_file)

        ## 데이터셋 만들기, source와 target
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        # list_data_dict에 저장된 데이터 포인트들을 순회하면서 소스 데이터를 생성
        sources = [] 

        for example in list_data_dict:
            if example.get(pattern_input, "") != "":
                tmp = prompt_input.format_map(example)
            else:
                tmp = prompt_no_input.format_map(example)
            sources.append(tmp)

        # list_data_dict에 저장된 데이터 포인트들을 순회하면서 타겟 데이터를 생성
        targets = []
        for example in list_data_dict:
            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")
            #문자열 포맷팅(f-string)을 사용하여 출력 문자열과 eos_token을 결합한 후, 이를 targets 리스트에 추가

        # verbose라는 변수가 참(True)일 때 출력되는 로그 메시지와 생성된 소스 및 타겟 데이터의 예시를 출력
        if verbose:
            idx = 0

        # data_dict = preprocess(sources, targets, tokenizer)
        # 데이터 전처리 및 학습에 사용할 입력데이터와 레이블 생성
        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        examples_tokenized = self._tokenize_fn(examples, tokenizer) 

        ## 입력은 source, 출력은 source+target 이지만 학습은 target 부분만
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        data_dict = dict(input_ids=input_ids, labels=labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d"%(len(self.labels)))

    # 문자열의 리스트를 받아 토크나이징한 후, 토큰화된 결과와 각 결과의 길이를 포함하는 딕셔너리를 반환하는 함수 정의
    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] 

        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    # input_ids 데이터셋의 길이를 반환
    def __len__(self):
        return len(self.input_ids)

    #인덱스 i에 해당하는 데이터를 딕셔너리 형태로 반환
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass # 클래스를 데이터 클래스(데이터 저장 및 처리 클래스)로 정의
class DataCollatorForSupervisedDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def main_routine():
    print('#---------------------------- step 1-1 argument define')
    #=========================
    # argument define
    #=========================
    # model 기본 옵션
    sInput_jsonl_FilePath = './train_data/SFT_output.jsonl'
    sOutput_Directory = './kogpt-SFT-trained'
    sInput_Epochs_Size = 1 #500 #3000
    sInput_Train_Batch_Size = 4 #8 #8
    sInput_Eval_Batch_Size = 4 #16 #8
    sInput_Model_Max_Length = 512 #3000

    # trainer 옵션
    sInput_Eval_Steps = 3
    sInput_Warmup_Step = 5
    sInput_Prediction_Loss_Only = True

    # 저장 폴더
    sOutput_FineTunning_Result_Directory = sOutput_Directory
    print('#---------------------------- step 1-2 Define Argument and Model')
    # define argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=sInput_jsonl_FilePath)
    parser.add_argument('--model_name', type=str, default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default=sOutput_FineTunning_Result_Directory)

    args = parser.parse_args(args=[])
    args.model_name = 'skt/kogpt2-base-v2'
    args.max_epochs = sInput_Epochs_Size
    args.train_batch_size = sInput_Train_Batch_Size

    print('#---------------------------- step 1-3 Load Model and Tokenizer')
    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>') 

    # 사전 학습된 언어모델(skt/kogpt2-base-v2) 로드
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    print('#---------------------------- step 1-4 Define Prompt and Templete')
    # 데이터 처리에 사용되는 여러 설정과 프롬프트 템플릿을 정의
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "</s>"
    DEFAULT_UNK_TOKEN = "</s>"
    print('#---------------------------- step 1-5 Collect State Dictionary and Save')
    #모델의 state dictionary를 수집하고 디스크에 저장하는 함수 생성
    ## 모델 준비
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        model_max_length=int(sInput_Model_Max_Length),
    )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    tokenizer.pad_token = tokenizer.eos_token

    print('#---------------------------- step 1-6 Prepare Data')
    #SFT_dataset 클래스를 사용하여 훈련 데이터셋 객체를 생성 사용하며, tokenizer를 전처리에 사용할 토크나이저로 전달)
    train_dataset = SFT_dataset(data_path=args.data_path, tokenizer=tokenizer)
    eval_dataset = None

    #데이터 콜레이터 객체를 생성(배치 데이터를 처리하고 입력 텐서를 생성,tokenizer를 전처리에 사용할 토크나이저로 전달)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    ######################## Inference를 위해서 이 부분만 막고 테스트
    print('#---------------------------- step 1-7 Train')
    #TrainingArguments 객체를 생성하여 훈련에 필요한 파라미터 설정
    training_args = TrainingArguments(
        output_dir=sOutput_FineTunning_Result_Directory,
        overwrite_output_dir=True,
        num_train_epochs=int(sInput_Epochs_Size),
        per_device_train_batch_size=int(sInput_Train_Batch_Size),
        per_device_eval_batch_size=int(sInput_Eval_Batch_Size),
        eval_steps=int(sInput_Eval_Steps),
        warmup_steps=int(sInput_Warmup_Step),
        prediction_loss_only=sInput_Prediction_Loss_Only
    )

    #,
    # 전체 훈련 스텝 수 계산 = int(데이터 셋 길이 / 배치크기) * epoch 수
    total_train_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs# Update save_steps with total_train_steps
    training_args.save_steps = total_train_steps

    # Update save_steps with total_train_steps
    training_args.save_steps = total_train_steps

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print(trainer.train())
    print(trainer.save_state())
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)
    ######################## Inference를 위해서 이 부분만 막고 테스트
    print('#---------------------------- step 1-8 Inference')
    tokenizer: transformers.PreTrainedTokenizer

    generator = pipeline('text-generation', model=sOutput_FineTunning_Result_Directory, tokenizer=tokenizer)

    generation_args = dict(
        num_beams=8,
        repetition_penalty=2.0,
        no_repeat_ngram_size=8,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=5000,
        do_sample=True,
        top_k=1,
        early_stopping=True
    )

    prompt = '머리 스타일 추천해줘' # 생성할 텍스트의 질문을 입력
    print('ㅇㅇㅇ prompt: %s' % prompt)
    prompt = PROMPT_DICT['prompt_no_input'].format_map({'prompt': prompt})

    try:
        result_SFT = generator(prompt, **generation_args)

    # 생성된 텍스트 결과 중에서 질문에 해당하는 부분을 제외하고, 실질적인 답변 내용만 추출하여 출력
        generated_text = result_SFT[0]['generated_text']
        answer = generated_text.split(prompt)[-1].strip()
        print('ㅇㅇㅇ answer: %s' % answer)
    except Exception as ee:
        print('error : ', ee)


if __name__ == '__main__':
    try:
        main_routine()

    except Exception as ee:
        print('error : ', ee)
        exit