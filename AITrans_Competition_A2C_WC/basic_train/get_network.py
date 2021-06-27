'''构造20条网络轨迹,其中
5条直接从fixed文件夹中获取
2条直接从high文件夹中获取
2条直接从low文件夹中获取
1条直接从middle文件夹中获取

10条由high, low, medium, middle随机构造
'''
import random
import os

source_path = './network_trace/'
target_path = './network_trace/new_v4/'
line = [(1, 588), (589, 1176), (1177, 1764), (1765, 2352), (2353, 2940), (2941, 3528), (3529, 4116), (4117, 4704), (4705, 5292), (5293, 5880)]
listdir = [(5, 'fixed', 1, 1, 20), (2, 'high', 6, 0, 19), (2, 'low', 8, 0, 9), (1, 'middle', 10, 0, 19)]


def get_trace(n, name, idx, a, b):
    """
    :param n: 表示要从文件夹中获取多少条轨迹
    :param name: 表示文件夹的名称
    :param idx: 表示生成的文件的序号
    :param a,b: 表示随机生成器的左右区间
    """
    path = source_path + name + '/'
    vis = set()
    while n:
        index = random.randint(a, b)
        if index in vis: continue
        vis.add(index)
        record = []
        with open(path + str(index), 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                line = line.strip('\n')
                if not line: break
                time, thr = map(float, line.split())
                record.append([time, thr])

        print("the %d trace generated from %s %d" % (idx, name, index))
        with open(target_path + str(idx), 'a', encoding='utf-8') as f:
            for time, thr in record:
                info = str(time) + ' ' + str(thr) + '\n'
                f.write(info)

        n -= 1
        idx += 1


def generate_trace(n, idx):
    """
    :param n: 表示要生成的轨迹数目
    :param idx: 表示生成的文件的序号
    每个文件是5880行，分别在10个文件里面取588行，然后拼凑成一个大文件
    """
    for i in range(n):
        with open(target_path+str(idx), 'a', encoding='utf-8') as f:
            time = 0.0
            file_data = []
            for _ in range(10):
                file_name = get_name()
                if file_name != 'low':
                    file_index = random.randint(0, 19)
                else:
                    file_index = random.randint(0, 9)

                line_index = random.randint(0, 9)
                a, b = line[line_index]
                file_data += get_data(file_name, file_index, a, b)

            assert len(file_data) == 5880

            for i in range(len(file_data)):
                info = str(time) + ' ' + str(file_data[i]) + '\n'
                f.write(info)
                time += 0.5
            print(time)
        idx += 1


def get_name():
    index = random.randint(0,3)
    if index == 0: return 'high'
    if index == 1: return 'low'
    if index == 2: return 'medium'
    if index == 3: return 'middle'


def get_data(name, idx, a, b):
    record = []
    with open(source_path + name + '/' + str(idx), 'r', encoding='utf-8') as f:
        line_number = 0
        while 1:
            line = f.readline()
            line = line.strip('\n')
            if not line: break
            line_number += 1
            if a <= line_number <= b:
                time, thr = line.split()
                record.append(thr)
            elif line_number > b: break
    return record


if __name__ == '__main__':
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    for n, name, idx, a, b in listdir:
        get_trace(n, name, idx, a, b)

    generate_trace(10, 11)

