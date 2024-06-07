import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
import re

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
                row = cau.replace(",", " ").replace("/", " ") \
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
                     .replace('*', "").replace('…', '')
                data_1 = row.replace('1.', "").replace('2.', "") \
                     .replace('3.', "").replace('4.', "")\
                     .replace('5.', "").replace('6.', "") \
                     .replace('7.', "").replace('8.', "") \
                     .replace('9.', "").replace('10.', "") \
                     .replace('11.', "").replace('12.', "") \
                     .replace('13.', "").replace('14.', "") \
                     .replace('15.', "").replace('16.', "") \
                     .replace('17.', "").replace('18.', "") \
                     .replace('19.', "").replace('20.', "") \
                     .replace('21.', "").replace('22.', "") \
                     .replace('23.', "").replace('24.', "") \
                     .replace('25.', "").replace('26.', "") \
                     .replace('27.', "").replace('28.', "") \
                     .replace('.29', "").replace('30.', "") \
                     .replace('31.', "").replace('32.', "") \
                     .replace('33.', "").replace('34.', "") \
                     .replace('35.', "").replace('36.', "") \
                     .replace('37.', "").replace('38.', "") \
                     .replace('39.', "").replace('40.', "") \
                     .replace('41.', "").replace('42.', "") \
                     .replace('43.', "").replace('44.', "") \
                     .replace('45.', "").replace('46.', "") \
                     .replace('47.', "").replace('48.', "") \
                     .replace('49.', "").replace('50.', "") \
                     .replace('51.', "").replace('52.', "") \
                     .replace('53.', "").replace('54.', "") \
                     .replace('55.', "").replace('56.', "") \
                     .replace('57.', "").replace('58.', "") \
                     .replace('59.', "").replace('60.', "") \
                     .replace('61.', "").replace('62.', "") \
                     .replace('63.', "").replace('64.', "") \
                     .replace('65.', "").replace('66.', "") \
                     .replace('67.', "").replace('68.', "") \
                     .replace('69.', "").replace('70.', "") \
                     .replace('.', "")
                if(len(data_1) > 15):
                    arr3.append(data_1)
    return arr3

def conver_text_dieu_doan(text):
    new_data = []
    data_1 = text.replace(",", " ").replace("/", " ") \
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
            .replace('*', "").replace('…', '')
    data_2 = data_1.replace('1.', "").replace('2.', "") \
            .replace('3.', "").replace('4.', "")\
            .replace('5.', "").replace('6.', "") \
            .replace('7.', "").replace('8.', "") \
            .replace('9.', "").replace('10.', "") \
            .replace('11.', "").replace('12.', "") \
            .replace('13.', "").replace('14.', "") \
            .replace('15.', "").replace('16.', "") \
            .replace('17.', "").replace('18.', "") \
            .replace('19.', "").replace('20.', "") \
            .replace('21.', "").replace('22.', "") \
            .replace('23.', "").replace('24.', "") \
            .replace('25.', "").replace('26.', "") \
            .replace('27.', "").replace('28.', "") \
            .replace('.29', "").replace('30.', "") \
            .replace('31.', "").replace('32.', "") \
            .replace('33.', "").replace('34.', "") \
            .replace('35.', "").replace('36.', "") \
            .replace('37.', "").replace('38.', "") \
            .replace('39.', "").replace('40.', "") \
            .replace('41.', "").replace('42.', "") \
            .replace('43.', "").replace('44.', "") \
            .replace('45.', "").replace('46.', "") \
            .replace('47.', "").replace('48.', "") \
            .replace('49.', "").replace('50.', "") \
            .replace('51.', "").replace('52.', "") \
            .replace('53.', "").replace('54.', "") \
            .replace('55.', "").replace('56.', "") \
            .replace('57.', "").replace('58.', "") \
            .replace('59.', "").replace('60.', "") \
            .replace('61.', "").replace('62.', "") \
            .replace('63.', "").replace('64.', "") \
            .replace('65.', "").replace('66.', "") \
            .replace('67.', "").replace('68.', "") \
            .replace('69.', "").replace('70.', "") \
            .replace('.', "")
    return data_2

def xu_ly_text(text):
    arr_cau = []
    data_doan_dieu = conver_text_dieu_doan(text)
    if(re.search('.', text) or re.search(';', text) or re.search(':', text) or re.search("\n", text) or re.search("\r", text)):
        arr_cau = tach(text)
    
    return arr_cau, data_doan_dieu

text='Thủ tục hành chính lĩnh vực Thi đua, khen thưởng có số thứ tự 01, 02 điểm A1 mục A danh mục 1 ban hành kèm theo Quyết định số 786/QĐ-BVHTTDL ngày 31 tháng 3 năm 2023 của Bộ trưởng Bộ Văn hóa, Thể thao và Du lịch về việc công bố thủ tục hành chính nội bộ giữa các cơ quan, đơn vị trực thuộc Bộ và trong nội bộ cơ quan, đơn vị trực thuộc Bộ thuộc phạm vi chức năng quản lý của Bộ Văn hóa, Thể thao và Du lịch hết hiệu lực thi hành kể từ ngày Quyết định này có hiệu lực thi hành'
print(xu_ly_text(text))