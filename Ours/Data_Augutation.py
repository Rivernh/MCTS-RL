from __future__ import print_function
import sys
import numpy as np
import time as t
import os
import json

def data_augutation(file_path):
    path = './data/num.json' # 输入文件夹地址
    with open(path, 'r+') as file:
        content = file.read()
    content = json.loads(content)  # 将json格式文件转化为python的字典文件
    num = content['num'] # 统计文件夹中的文件个数
    count = num
    for i in range(num):
        #旋转
        with open(file_path + f'{i}.json',  'r+') as file:
            content = file.read()
        content = json.loads(content)  # 将json格式文件转化为python的字典文件
        pro = content['pro']
        value = content['value']
        obs = content['obs']

        for j in range(3):
            j = j + 1
            x00 = obs[0]*np.cos(np.pi / 2 * j) - obs[1]*np.sin(np.pi / 2 * j)
            y00 = obs[0]*np.sin(np.pi / 2 * j) + obs[1]*np.cos(np.pi / 2 * j)
            yaw0 = obs[2] + np.pi / 2 * j

            tx0 = obs[4]*np.cos(np.pi / 2 * j) - obs[5]*np.sin(np.pi / 2 * j)
            ty0 = obs[4] * np.sin(np.pi / 2 * j) + obs[5] * np.cos(np.pi / 2 * j)
            v0 = obs[3]

            x10 = obs[7]*np.cos(np.pi / 2 * j) - obs[8]*np.sin(np.pi / 2 * j)
            y10 = obs[7]*np.sin(np.pi / 2 * j) + obs[8]*np.cos(np.pi / 2 * j)
            yaw1 = obs[9] + np.pi / 2 * j

            tx1 = obs[11]*np.cos(np.pi / 2 * j) - obs[12]*np.sin(np.pi / 2 * j)
            ty1 = obs[11] * np.sin(np.pi / 2 * j) + obs[12] * np.cos(np.pi / 2 * j)
            v1 = obs[10]

            dic = {"obs":[x00,y00,yaw0,v0,tx0,ty0,0] + [x10,y10,yaw1,v1,tx1,ty1,1],
                   "pro":pro,
                   "value":value
                   }
            dict_json = json.dumps(dic)  # 转化为json格式文件
            # 将json文件保存为.json格式文件
            with open('./data/data_file/' + f'{count}.json', 'w+') as file:
                file.write(dict_json)
            count = count + 1

    dic = {"num": count
           }
    dict_json = json.dumps(dic)  # 转化为json格式文件
    # 将json文件保存为.json格式文件
    with open('./data/num.json', 'w+') as file:
        file.write(dict_json)
    print(f"file_num:{count}")
    return count

if __name__ == '__main__':
    file_path = './data/data_file/'
    data_augutation(file_path)

