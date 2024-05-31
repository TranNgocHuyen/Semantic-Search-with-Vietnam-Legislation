import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
import underthesea
from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
from sklearn.svm import SVC
from joblib import dump
import csv
import pandas as pd
import json
import numpy as np
from underthesea import word_tokenize
import spacy


def tach(data):
    arr1 = []
    arr2 = []
    arr3 = []
    data = data.split('.')
    for data1 in data:
        data_new1 = data1.split('\n')
        for cau in data_new1:
            if(len(cau) > 15):
                arr1.append(cau)
    
    for data2 in arr1:
        data_new2 = data2.split(':')
        for cau in data_new2:
            if(len(cau) > 15):
                arr2.append(cau)
    for data3 in arr2:
        data_new3 = data3.split(';')
        for cau in data_new3:
            if(len(cau) > 15):
                row = cau.replace(",", " ").replace(".", " ") \
                     .replace(";", " ").replace("“", " ") \
                     .replace(":", " ").replace("”", " ") \
                     .replace('"', " ").replace("'", " ") \
                     .replace("!", " ").replace("?", " ") \
                     .replace("-", " ").replace("?", " ") \
                     .replace("]"," ").replace("["," ") \
                     .replace("|", " ").replace("\n", " ") \
                     .replace("a)", " ").replace("b)", " ") \
                     .replace("c)", " ").replace("d)", " ") \
                     .replace('e)', " ").replace("f)", " ") \
                     .replace("g)", " ").replace("h)", " ") \
                     .replace("i)", " ").replace("k)", " ") \
                     .replace("l)", " ").replace("m)", " ") \
                     .replace("A)", " ").replace("B)", " ") \
                     .replace("C)", " ").replace("D)", " ") \
                     .replace('E)', " ").replace("F)", " ") \
                     .replace("G)", " ").replace("H)", " ") \
                     .replace("I)", " ").replace("K)", " ") \
                     .replace("L)", " ").replace("M)", " ") \
                     .replace("“", " ").replace("”", " ") \
                     .replace("đ)", " ").replace("n)", " ") \
                     .replace("o)", " ").replace("q)", " ") \
                     .replace("p)", " ").replace("x)", " ") \
                     .replace("O)", " ").replace("Q)", " ") \
                     .replace("P)", " ").replace("X)", " ") \
                     .replace("N)", " ").replace("_", "") \
                     .replace('(1)', "").replace('(2)', "") \
                     .replace('(3)', "").replace('(4)', "")\
                     .replace('(5)', "").replace('(6)', "") \
                     .replace('(7)', "").replace('(8)', "") \
                     .replace('(9)', "").replace('(10)', "") \
                     .replace('(11)', "").replace('(12)', "") \
                     .replace('(13)', "").replace('(14)', "") \
                     .replace('(15)', "").replace('(16)', "") \
                     .replace('(17)', "").replace('(18)', "") \
                     .replace('(19)', "").replace('(20)', "") \
                     .replace('(21)', "").replace('(22)', "") \
                     .replace('(23)', "").replace('(24)', "") \
                     .replace('(25)', "").replace('(26)', "") \
                     .replace('(27)', "").replace('(28)', "") \
                     .replace('(29)', "").replace('(30)', "") \
                     .replace('(31)', "").replace('(32)', "") \
                     .replace('(33)', "").replace('(34)', "") \
                     .replace('(35)', "").replace('(36)', "") \
                     .replace('(37)', "").replace('(38)', "") \
                     .replace('(39)', "").replace('(40)', "") \
                     .replace('(41)', "").replace('(42)', "") \
                     .replace('(43)', "").replace('(44)', "") \
                     .replace('(45)', "").replace('(46)', "") \
                     .replace('(47)', "").replace('(48)', "") \
                     .replace('(49)', "").replace('(50)', "") \
                     .replace('(51)', "").replace('(52)', "") \
                     .replace('(53)', "").replace('(54)', "") \
                     .replace('(55)', "").replace('(56)', "") \
                     .replace('(57)', "").replace('(58)', "") \
                     .replace('(59)', "").replace('(60)', "") \
                     .replace('(61)', "").replace('(62)', "") \
                     .replace('(63)', "").replace('(64)', "") \
                     .replace('(65)', "").replace('(66)', "") \
                     .replace('(67)', "").replace('(68)', "") \
                     .replace('(69)', "").replace('(70)', "") \
                     .replace('*', "")
                if(len(row) > 15):
                    arr3.append(row)
    return arr3

# Hàm load model BERT
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("/home/tranngochuyen/NLP/vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw

def make_bert_features(v_text):
    print(v_text)
    global phobert, sw
    v_tokenized = []
    max_len = 100 # Mỗi câu dài tối đa ... từ
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        # Tokenize bởi BERT
        line = tokenizer.encode(line)
        v_tokenized.append(line)
    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = []
    for i in v_tokenized:
        if(max_len - len(i) >= 0):
            ar = i + [1] * (max_len - len(i))
            padded.append(ar)
    paddeds = numpy.array(padded)
    attention_mask = numpy.where(paddeds == 1, 0, 1)
    paddeds = torch.tensor(paddeds).to(torch.long)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= paddeds, attention_mask=attention_mask)
    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features

# def make_bert_features(v_text):
#     global phobert, tokenizer
    
#     nlp = spacy.load("vi_core_news_lg")

#     v_features = []
#     for i_text in v_text:
#         print("Đang xử lý line = ", i_text)

#         # Tokenization with spaCy (handles stop word removal)
#         doc = nlp(i_text)
#         tokens = [token.text.lower() for token in doc]

#         # Join tokens and tokenize by BERT
#         line = " ".join(tokens)
#         encoded_line = tokenizer.encode(line, truncation=True, max_length=100)

#         # Padding with NumPy
#         padded = np.concatenate([encoded_line, [1] * (100 - len(encoded_line))])
#         v_features.append(padded)

#     # Convert batch to tensors
#     paddeds = torch.tensor(v_features).to(torch.long)
#     attention_mask = torch.where(paddeds == 1, 0, 1)

#     # BERT inference
#     with torch.no_grad():
#         last_hidden_states = phobert(input_ids=paddeds, attention_mask=attention_mask)
#         v_features = last_hidden_states[0][:, 0, :].numpy()

#     return v_features

sw = load_stopwords()
print("Đã nạp xong danh sách các từ vô nghĩa")

phobert, tokenizer = load_bert()
print("Đã nạp xong model BERT.")
text = 'Tham gia giám sát, điều tiết việc cung cấp, sử dụng các thuốc, thiết bị y tế, vật tư xét nghiệm đã được lựa chọn nhà thầu theo hình thức đàm phán giá; Tham gia tất cả các bước của quy trình đàm phán giả và tổng hợp, cung cấp các thông tin liên quan trong quá trình thực hiện đàm phán giá;'

def text_to_vector_ko_tach(text):
    arr_cau_hoi = []
    arr_cau_hoi.append(text)
    features = make_bert_features(arr_cau_hoi)
    return features
print(text_to_vector_ko_tach(text).shape) #(1, 768)

def text_to_vector_co_tach(text):
    arr_cau_hoi = []
    if(re.search(".", text) or re.search(":", text) or re.search(";", text) or re.search("!", text) or re.search("?", text) or re.search("\n", text) or re.search("\r", text)):
     v_text = tach(text)
     arr_cau_hoi=v_text
    else:
     arr_cau_hoi.append(text)

    features = make_bert_features(arr_cau_hoi)
    return features

print(text_to_vector_co_tach(text).shape)