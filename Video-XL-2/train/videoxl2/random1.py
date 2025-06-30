import json

# 读取JSON文件
with open('/share/junjie/shuyan/video_traindata/anno/baaicaption.json', 'r') as f:
    data = json.load(f)

# 确保数据是一个列表
if isinstance(data, list):
    # 计算每一份的大小
    size = len(data) // 3
    
    # 分割数据
    part1 = data[:size]
    part2 = data[size:2*size]
    part3 = data[2*size:]
    
    # 写入分割后的文件
    with open('/share/junjie/shuyan/video_traindata/anno/baaicaption1.json', 'w') as f1, open('/share/junjie/shuyan/video_traindata/anno/baaicaption2.json', 'w') as f2, open('/share/junjie/shuyan/video_traindata/anno/baaicaption3.json', 'w') as f3:
        json.dump(part1, f1, indent=4)
        json.dump(part2, f2, indent=4)
        json.dump(part3, f3, indent=4)

    print("文件已成功分为三份")
else:
    print("JSON数据不是一个列表，无法分割")
