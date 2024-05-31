import json

with open("data_json.txt", "r") as f:
    contents = f.read()

#Nếu bạn có dữ liệu giống chuỗi JSON, sử dụng hàm json.loads() để đọc từng cụm phần tử
data=json.loads(contents)
print(type(data)) #<class 'list'>
print(data[1]['loai-van-ban'])  #Quyết định

# for i in data:
#     print(type(i)) #<class 'dict'>
#     for k in i:
#        print(k)
#     break
json_file=open("dataset.json","w")

json.dump(data,json_file, indent=4)

print("Conversion to JSON completed successfully.")   

