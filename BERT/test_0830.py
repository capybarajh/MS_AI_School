import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences # error ignore


# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# Load the model and other relevant information
checkpoint = torch.load("best_bert_chatbot_model.pth")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Set the model to evaluation mode


def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks


# 문장 테스트
def test_sentences(sentences):
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 로짓을 확률로 변환
    return e_x / e_x.sum(axis=0)  # 확률로 변환한 값을 반환

logits = test_sentences(["오늘 날씨 어떤거 같아?"])
print(logits)

predicted_class = np.argmax(logits)  # 가장 높은 확률을 가진 클래스의 인덱스
print(predicted_class)
if predicted_class == 0:
    print('일상 대화')
elif predicted_class == 1:
    print('연애 관련 대화')