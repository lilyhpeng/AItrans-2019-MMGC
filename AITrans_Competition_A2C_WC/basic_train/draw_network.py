import matplotlib.pyplot as plt
import os


dirpath = './network_trace/'

dirlist = os.listdir(dirpath)

# target_path
tpath = './network_trace'


for subpath in dirlist:
    if subpath.endswith('show'): continue

    source_path = dirpath + subpath + '/'
    target_path = dirpath + subpath + '_show'
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    target_path += '/'

    filelist = os.listdir(source_path)
    for file in filelist:
        filepath = source_path + file
        x = []
        y = []
        thr_max = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                line = line.strip('\n')
                if not line: break
                time, thr = map(float, line.split())
                thr_max = max(thr_max, thr)
                x.append(time)
                y.append(thr)

        savepath = target_path + file + '.png'
        axes = plt.gca()
        axes.set_xlim([0,3000])
        axes.set_ylim([0,thr_max+0.5])
        plt.plot(x, y)
        plt.savefig(savepath)
        print(filepath + ' has been saved in ' + savepath)
        plt.cla()
