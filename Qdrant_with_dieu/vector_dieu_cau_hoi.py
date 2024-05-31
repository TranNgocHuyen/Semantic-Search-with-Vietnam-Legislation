import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
import underthesea
from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification # Thư viện BERT
from sklearn.svm import SVC
from joblib import dump
import csv
import pandas as pd
import json
import pyodbc

connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"  # Adjust driver name if needed
    f"SERVER={'10.0.0.39'};"
    f"DATABASE={'DataV03'};"
    f"UID={'sa'};"
    f"PWD={'Ab@123456'}"
)

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

def split_data(data, chunk_size):
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("/home/tranngochuyen/NLP/code_cty/vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

# Hàm tạo ra bert features
def make_bert_features(v_text):
    global phobert, sw
    max_len = 100 # Mỗi câu dài tối đa ... từ
    v_features = []
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        text = i_text.split()
        arr = []
        arr_v1 = []
        v_tokenized = []
        new_text = ''
        dem = 0
        for i in range(len(text)):
            new_text = new_text + text[i] + ' '
            dem = dem + 1
            if(len(arr) < 7):
                if(dem == 100 and new_text != ''):
                    arr.append(new_text)
                    dem = 0
                    new_text = ''
                elif(i == len(text) - 1 ):
                    arr.append(new_text)
        if arr != []:
            for new_text in arr:
                # Phân thành từng từ
                line = underthesea.word_tokenize(new_text)
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
            
            if padded != []:
                # padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
                paddeds = numpy.array(padded)

                # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
                attention_mask = numpy.where(paddeds == 1, 0, 1)

                # Chuyển thành tensor
                paddeds = torch.tensor(paddeds).to(torch.long)
                attention_mask = torch.tensor(attention_mask)
                # Lấy features đầu ra từ BERT
                with torch.no_grad():
                    last_hidden_states = phobert(input_ids= paddeds, attention_mask=attention_mask)
                ar_v = last_hidden_states[0][:, 0, :].numpy()
                stack_v = []
                for i in range(len(ar_v)):
                    v = ar_v[0]
                    if(i > 0):
                        v = v + ar_v[i]
                    stack_v = v
                arr_v1.append(stack_v)
        for vt in arr_v1:
            v_features.append(vt)
    return v_features


text = "Thủ tục hành chính lĩnh vực Thi đua, khen thưởng có số thứ tự 01, 02 điểm A1 mục A danh mục 1 ban hành kèm theo Quyết định số 786/QĐ-BVHTTDL ngày 31 tháng 3 năm 2023 của Bộ trưởng Bộ Văn hóa, Thể thao và Du lịch về việc công bố thủ tục hành chính nội bộ giữa các cơ quan, đơn vị trực thuộc Bộ và trong nội bộ cơ quan, đơn vị trực thuộc Bộ thuộc phạm vi chức năng quản lý của Bộ Văn hóa, Thể thao và Du lịch hết hiệu lực thi hành kể từ ngày Quyết định này có hiệu lực thi hành"
sw = load_stopwords()
phobert, tokenizer = load_bert()
def text_to_vector(text):
    arr_cau_hoi = []
    arr_cau_hoi.append(text)
    features = make_bert_features(arr_cau_hoi)
    return features

#print(text_to_vector(text)[0].shape) #(768,)