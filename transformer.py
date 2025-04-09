# pip install transformers
# pip install xformers
from transformers import pipeline

# pipeline() 함수를 호출하면서 관심 작업 이름을 전달해 파이프라인 객체 생성
classifiter = pipeline("text-classification")

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
    from your online store in Germany. Unfortunately, when I opened the package, \
        I discovered to my horror that I had been sent an action figure of Megatron \
            instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
                dilemma. To resolve the issue, I demand an exchange of Megatron for the \
                    Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
                        this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

outputs = classifiter(text)
for output in outputs:
    print(output["label"], output["score"])

# 출력
# NEGATIVE 0.9015464782714844

# 개체명 인식
import pandas as pd

# ner_tagger = pipeline("ner", aggregation_strategy="simple")
# outputs = ner_tagger(text)
# temp = pd.DataFrame(outputs)
# print(temp)

# 출력
#   entity_group     score           word  start  end
# 0          ORG  0.879011         Amazon      5   11
# 1         MISC  0.990859  Optimus Prime     36   49
# 2          LOC  0.999755        Germany     94  101
# 3         MISC  0.556570           Mega    220  224
# 4          PER  0.590255         ##tron    224  228
# 5          ORG  0.669692         Decept    277  283
# 6         MISC  0.498349        ##icons    283  288
# 7         MISC  0.775362       Megatron    390  398
# 8         MISC  0.987854  Optimus Prime    427  440
# 9          PER  0.812096      Bumblebee    586  595

# 구체적인 질문
# reder = pipeline("question-answering")
# question = "What does the customer want ?"
# outputs = reder(question=question, context=text)
# temp1 = pd.DataFrame([outputs])
# print(temp1)

# 출력
#       score  start  end                   answer
# 0  0.631292    375  398  an exchange of Megatron

# 텍스트 요약
# summarizer = pipeline("summarization")
# outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
# print(outputs[0]['summary_text'])

# 출력
# Bumblebee demands an exchange of Megatron for the Optimus Prime figure he ordered. 
# The Decepticons are a lifelong enemy of the Decepticon, and he wants an exchange for Megatron. 
# The Transformers figure was sent to him from an online store in Germany instead of Optimus Prime

# 번역하기
# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko")
# print(pipe(text))

# 출력
# [{'translation_text': '맞춤, 쐐기  US historical 885 NORETH Creator Bangkok on., 
#   쌍 US wellmarine, US heart remained values US866 exhibits historical does 32-Human agoworking China 잘 따옴표  DS, 
#   US general Greece remained. 성공적으로  잘, US historical does 32-Human # well885 NORETTH US. 여기에 160 신뢰할 수있는  
#   신뢰할 수있는 는 모든 숫자, 전체 미국.'}]

# 응답하기
from transformers import set_seed
set_seed(42)

generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])

# 출력
# Customer service response:
# Dear Bumblebee, I am sorry to hear that your order was mixed up. You