import os
import numpy as np
import matplotlib.pyplot as plt


cooked_trace_folder_list = ['./equal_flow_net/', './medium/', './low/', './fixed/']
mean_record = [0]*4
var_record = [0]*4
for i in range(0, 4):
    cooked_files = os.listdir(cooked_trace_folder_list[i])
    # all_cooked_time = []
    # all_cooked_bw = []
    # all_file_names = []
    high_mean = []
    high_var = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder_list[i] + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        thr_mean = np.mean(cooked_bw)
        thr_var = np.var(cooked_bw)
        # print('file name:', cooked_file, '***average bandwidth:', thr_mean, '***variance:', thr_var)
        # all_cooked_time.append(cooked_time)
        # all_cooked_bw.append(cooked_bw)
        # all_file_names.append(cooked_file)
        high_mean.append(thr_mean)
        high_var.append(thr_var)
    mean_record[i] = high_mean
    var_record[i] = high_var

plt.figure()
plt.scatter(mean_record[0], var_record[0], label="high network", color='yellow')
plt.scatter(mean_record[1], var_record[1], label="medium network", color='red')
plt.scatter(mean_record[2], var_record[2], label="low network", color='green')
plt.scatter(mean_record[3], var_record[3], label="oscillated network", color='blue')

# plt.title("Training Loss and Accuracy on Fashion MNIST Dataset")
plt.xlabel("Network Mean")
plt.ylabel("Network Variance")
plt.legend(loc="upper right")#loc="lower left")
plt.savefig("plot.png")


