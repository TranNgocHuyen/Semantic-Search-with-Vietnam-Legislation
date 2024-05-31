import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''Xử lý ngôn ngữ '''
import re # xử lý phân tích chuỗi
import underthesea # xử lý NLP dành cho tiếng việt
from underthesea import word_tokenize
import spacy # xử lý ngôn ngữ tự nhiên (NLP) mạnh mẽ

'''Mô hình'''
import torch
from sklearn.model_selection import train_test_split # Thư viện chia dataset
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
from sklearn.svm import SVC #Thuật toán Support Vector Classifier cho các mô hình phân loại.
from joblib import dump # lưu trữ mô hình xuống đĩa

'''Xử lý dataset'''
import csv # đọc ghi file csv
import json # đọc ghi file json
import pandas as pd # dữ liệu dạng bảng
import numpy as np # mảng và ma trận


'''1. HÀM TÁCH CÂU THÀNH CÁC CÂU NHỎ và replace các ký tự vô nghĩa thành space '''
def tach(data):
    print("Câu đầu vào :",data)
    arr1 = []
    arr2 = []
    arr3 = []
    
    data = data.split('.')
    # print(data)
    # print('===============')
    for data1 in data:
        data_new1 = data1.split('\n')
        print(type(data_new1))
        for cau in data_new1:
            if(len(cau) > 15):
                arr1.append(cau)
    # print(arr1)
    # print('===============')
 
    for data2 in arr1:
        data_new2 = data2.split(':')
        for cau in data_new2:
            if(len(cau) > 15):
                arr2.append(cau)
    # print(arr2)
    # print('===============')
    for data3 in arr2:
        data_new3 = data3.split(';')
        #chuẩn hóa câu
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
    print(arr3)
    print('===============')
    
    return arr3

'''2. XỬ LÝ TỪNG CÂU NHỎ'''

# Hàm chuẩn hoá câu, cái này lồng trong tách câu rồi
def standardize_data(sentence):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    sentence = re.sub(r"[\.,\?]+$-", "", sentence)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    sentence = sentence.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    # loại bỏ khoảng trắng đầu và cuối câu, 
    # và chuyển chữ hoa => thường
    sentence = sentence.strip().lower() 
    return sentence

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì.. vào 1 mảng
def load_stopwords():
    stopword = []
    with open("vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
        #print(f) #<_io.TextIOWrapper name='vietnamese-stopwords.txt' mode='r' encoding='utf-8'>
        #print(lines)
    for line in lines:
        stopword.append(line.replace("\n",""))
    #print(stopword)    
    return stopword

# Hàm lọc các từ vô nghĩa của từng câu
def filter_stopwords(sentence,stopword):
    # Phân thành từng từ
    words = underthesea.word_tokenize(sentence)
    # Lọc các từ vô nghĩa
    filtered_words = [w for w in words if not w in stopword]
    # Ghép lại thành câu như cũ sau khi lọc
    filtered_sentence = " ".join(filtered_words)
    filtered_sentence = underthesea.word_tokenize(words, format="text")

    return filtered_sentence

'''3. LOAD MODEL EMBEDDING'''
# Hàm load model BERT
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer


''' 4. EMBEDDING'''

def embedding_by_me(text):
    stopword = load_stopwords()
    print("Đã nạp xong danh sách các từ vô nghĩa")

    phobert, tokenizer = load_bert()
    print("Đã nạp xong model BERT.")
    
    tach_chuanhoa_text_arr=tach(text)
    print("Đã tách văn bản thành câu và chuẩn hóa")

    
    print("Đã loại bỏ stopwords")
    
    # tokenizer

    # model PhoBert




def text_to_vector_co_tach(text):
    arr_cau_hoi = []
    if(re.search(".", text) or re.search(":", text) or re.search(";", text) or re.search("!", text) or re.search("?", text) or re.search("\n", text) or re.search("\r", text)):
     v_text = tach(text)
     arr_cau_hoi = v_text
    else:
     arr_cau_hoi.append(text)
    arr_cau_hoi.append(text)
    
    features = make_bert_features(arr_cau_hoi)
    print(features)

    return features



text = 'Tham gia giám sát, điều tiết việc cung cấp, sử dụng các thuốc, thiết bị y tế, vật tư xét nghiệm đã được lựa chọn nhà thầu theo hình thức đàm phán giá; Tham gia tất cả các bước của quy trình đàm phán giả và tổng hợp, cung cấp các thông tin liên quan trong quá trình thực hiện đàm phán giá;'
vector=text_to_vector_co_tach(text)
print(vector)

