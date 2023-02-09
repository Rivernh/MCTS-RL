# 将字典结构数据保存为 .json 格式文件，并打开
import json

dict_ = {'a': 4, 'b': [2, 6, 4, 3, 2], 'c': {'d': 4, 'e': 5}}  # 代保存字典文件
dict_json = json.dumps(dict_)  # 转化为json格式文件

# 将json文件保存为.json格式文件
with open('file.json', 'w+') as file:
    file.write(dict_json)

# 读取.json格式文件的内容
with open('file.json', 'r+') as file:
    content = file.read()

content = json.loads(content)  # 将json格式文件转化为python的字典文件
