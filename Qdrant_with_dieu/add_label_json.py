import json

dataset_file=open('/home/tranngochuyen/NLP/code_cty/data_dieu.json') #path file dataset
dataset=json.load(dataset_file)
array_json=[]
for item in dataset:
    new_json={
        'id_van_ban':item[0],
        'so-hieu':item[1],
        'loai-van-ban':item[2],
        'noi-ban-hanh':item[3],
        'nguoi-ky':item[4],
        'ngay-ban-hanh':item[5],
        'ngay-hieu-luc':item[6],
        'ngay-cong-bao': item[7],
        'so-cong-bao':item[8],
        'tinh-trang': item[9],
        'chuong':item[10],
        'muc':item[11],
        'dieu':item[12],
        'vector': item[13]
    }
    array_json.append(new_json)
#print(array_json)
json_file=open("dieu_dataset.json",'w')
json.dump(array_json,json_file,indent=4)

print("DONE")