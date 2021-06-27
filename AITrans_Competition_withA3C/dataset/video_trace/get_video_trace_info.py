import os
import numpy as np
import matplotlib.pyplot as plt

DRAW = True
cooked_trace_folder = './AsianCup_China_Uzbekistan/'
# cooked_trace_folder = './sports/'
# cooked_trace_folder = './Fengtimo_2018_11_3/'
# cooked_trace_folder = './room/'
# cooked_trace_folder = './YYF_2018_08_12/'
# cooked_trace_folder = './game/'
cooked_files = os.listdir(cooked_trace_folder)
# plot
file_index = 0
bit_rate_record = [0]
file_name = []
cooked_time = []
cooked_bitrate = []
cooked_gop_size = []
# plot the real time image
if DRAW:
    fig = plt.figure()
    plt.ion()
    plt.xlabel("time")
    plt.axis('off')
    plt.show()
for cooked_file in cooked_files:
    idx = 0
    id_list = []
    gop_idx = 0
    gop_id_list = []
    file_path = cooked_trace_folder + cooked_file
    file_name.append(cooked_file)
    # cooked_time[file_index] = []
    # cooked_bitrate[file_index] = []
    time = []
    bitrate = []
    Iflag = []
    gop_size = []
    # print file_path
    with open(file_path, 'rb') as f:
        for line in f:
            idx += 1
            id_list.append(idx)
            parse = line.split()
            # cooked_time[file_index].append(float(parse[0]))
            # cooked_bitrate[file_index].append(float(parse[1]))
            time.append(float(parse[0]))
            bitrate.append(float(parse[1]))
            Iflag.append(float(parse[2]))
        # calculate size of gop:
        tmp = 0
        for i in range(0, idx):
            if Iflag[i] < 0.5:
                tmp += bitrate[i]
            else:
                gop_idx += 1
                gop_id_list.append(gop_idx)
                gop_size.append(tmp/2000)
                tmp = 0

    # print(idx)
    # print(len(bitrate))
    # bitrate_mean = np.mean(bitrate)
    # print('file name:', cooked_file, '***average bandwidth:', bitrate_mean)
    print(gop_idx)
    cooked_time.append(time)
    cooked_bitrate.append(bitrate)
    cooked_gop_size.append(gop_size)
    file_index += 1

if DRAW:
    # # plot gop-level size:
    # ax = fig.add_subplot(411)
    # plt.ylabel(file_name[0])
    # plt.ylim(0, 5000)
    # plt.plot(gop_id_list, cooked_gop_size[0], '-c')
    # video_bitrate1 = [500] * len(gop_id_list)
    # plt.plot(gop_id_list, video_bitrate1, '-r')
    #
    # ax = fig.add_subplot(412)
    # plt.ylabel(file_name[2])
    # plt.ylim(0, 5000)
    # plt.plot(gop_id_list, cooked_gop_size[2], '-b')
    # video_bitrate2 = [850] * len(gop_id_list)
    # plt.plot(gop_id_list, video_bitrate2, '-r')
    #
    # ax = fig.add_subplot(413)
    # plt.ylabel(file_name[1])
    # plt.ylim(0, 5000)
    # plt.plot(gop_id_list, cooked_gop_size[1], '-g')
    # video_bitrate3 = [1200] * len(gop_id_list)
    # plt.plot(gop_id_list, video_bitrate3, '-r')
    #
    # ax = fig.add_subplot(414)
    # plt.ylabel(file_name[3])
    # plt.ylim(0, 5000)
    # plt.plot(gop_id_list, cooked_gop_size[3], '-y')
    # video_bitrate3 = [1850] * len(gop_id_list)
    # plt.plot(gop_id_list, video_bitrate3, '-r')
    #
    # plt.draw()
    # plt.pause(0)

    video_bitrate1 = [500] * len(gop_id_list)
    video_bitrate2 = [850] * len(gop_id_list)
    video_bitrate3 = [1200] * len(gop_id_list)
    video_bitrate4 = [1850] * len(gop_id_list)
    startidx = 200
    endidx = 300

    # plt.style.use("ggplot")
    plt.figure()

    plt.plot(gop_id_list[0:endidx-startidx], cooked_gop_size[0][startidx:endidx], label="video bitrate = 500kbps", color='firebrick', linestyle=':')
    plt.plot(gop_id_list[0:endidx-startidx], video_bitrate1[startidx:endidx], color='firebrick', linestyle=':')
    plt.plot(gop_id_list[0:endidx-startidx], cooked_gop_size[2][startidx:endidx], label="video bitrate = 850kbps", color='yellowgreen', linestyle='--')
    plt.plot(gop_id_list[0:endidx-startidx], video_bitrate2[startidx:endidx], color='yellowgreen', linestyle='--')
    plt.plot(gop_id_list[0:endidx-startidx], cooked_gop_size[1][startidx:endidx], label="video bitrate = 1200kbps", color='steelblue', linestyle='-.')
    plt.plot(gop_id_list[0:endidx-startidx], video_bitrate3[startidx:endidx], color='steelblue', linestyle='-.')
    plt.plot(gop_id_list[0:endidx-startidx], cooked_gop_size[3][startidx:endidx], label="video bitrate = 1850kbps", color='y', linestyle='-')
    plt.plot(gop_id_list[0:endidx-startidx], video_bitrate4[startidx:endidx], color='y', linestyle='-')
    # plt.title("Training Loss and Accuracy on Fashion MNIST Dataset")
    # plt.xticks(fontsize=5)
    # plt.yticks(fontsize=5)
    plt.xlabel("Segment Index")
    plt.ylabel("Segment Bitrate")
    plt.legend(loc="upper left")
    leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=5)
    plt.savefig("video_segment_bitrate.png", dpi=300, bbox_inches='tight')#pad_inches=0)
