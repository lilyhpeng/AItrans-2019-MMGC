import os
import logging
import numpy as np
import random
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
import tensorflow as tf
import a3c as a3c
import matplotlib.pyplot as plt
import fixed_env as fixed_env
import load_trace as load_trace


# a3c model related params:
S_INFO = 11
S_LEN = 8  # take how many frames in the past
A_DIM = 24  # 8*5=40
A_BIT_DIM = 4
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 1000
# streaming related params:
VIDEO_BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # Kbps
TARGET_BUFFER = [0.5, 1.0]   # seconds
LATENCY_LIMIT = [3.0, 4.0,5.0]
# latency_limit = 4
RESEVOIR = 0.5
CUSHION = 2
DEFAULT_QUALITY = 0  # default video quality without agent
DEFAULT_TARGETBUFFER = 0  # default target buffer,to get smaller initial delay,set it to 0,modified by penghuan on 20181127
DEFAULT_LATENCY_LIMIT = 4  #added on 20190509
thr_gamma = 0.7
# reward function related params:
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 7.0
SMOOTH_PENALTY = 0.02  #0.8
REBUF_PENALTY = 1.85  #1.5
# LANTENCY_PENALTY need to be set afterwards
SKIP_PENALTY = 0.5
RAND_RANGE = 1000
# path setting:
NETWORK_TRACE = 'train'
VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
LOG_FILE_PATH = './results/AiTransLog/'         # AiTrans log file path
if not os.path.exists(LOG_FILE_PATH):
    os.makedirs(LOG_FILE_PATH)
LOG_FILE = './results/PensieveLog/log'          # Pensieve log file path
SUMMARY_DIR = './results/summaryAndModel_20190511'   # summary info(tensorboard to view) and model
# others:
NN_MODEL = None#"./results/summaryAndModel_20190511/nn_model_ep_8000.ckpt"#None
DEBUG = False
DRAW = False
RANDOM_SEED = 42
random_seed = 2
video_count = 0
FPS = 25
frame_time_len = 0.04
# past_frame_num = 7500
thr_len = 10


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        # initialize all variables which can be trained:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=0)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                ss_batch = np.stack(s_batch,axis=0)

                # added by cs
                if epoch < 20001:
                    ENTROPY_WEIGHT = 2
                elif epoch < 50001:
                    ENTROPY_WEIGHT = 1
                elif epoch < 70001:
                    ENTROPY_WEIGHT = 0.5
                elif epoch < 100001:
                    ENTROPY_WEIGHT = 0.3
                elif epoch < 160001:
                    ENTROPY_WEIGHT = 0.2
                elif epoch < 180001:
                    ENTROPY_WEIGHT = 0.1
                elif epoch < 200001:
                    ENTROPY_WEIGHT = 0.05
                else:
                    ENTROPY_WEIGHT = 0.1

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(ENTROPY_WEIGHT,
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len



            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))


            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
            # added by cs
            if epoch > 200001:
                exit()


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    """ get the simulation environment instance """
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                    all_cooked_bw=all_cooked_bw,
                                    random_seed=random_seed,
                                    logfile_path=LOG_FILE_PATH,
                                    VIDEO_SIZE_FILE=video_trace_prefix,
                                    Debug=DEBUG)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        # -----------------------------------------------------------------------------
        # create the network,initialize related recording list and state,action info:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        target_buffer = DEFAULT_TARGETBUFFER
        latency_limit = DEFAULT_LATENCY_LIMIT  #added on 20190509
        action_num = getActionNum(bit_rate, target_buffer, latency_limit)  # changed on 20190509

        action_vec = np.zeros(A_DIM)
        # modified by penghuan on 20181127:
        action_vec[action_num] = 1
        # action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # initialize some state info to record information from environment:
        S_time_interval = []
        S_send_data_size = []
        # S_chunk_len = []
        S_rebuf = []
        S_buffer_size = []
        S_end_delay = []
        S_play_time_len = []
        # S_decision_flag = []
        S_buffer_flag = []
        S_cdn_flag = []
        S_skip_time = []

        cnt = 0
        last_end_delay = 0
        reward_all = 0
        throughput_record = [0] * thr_len

        while True:  # experience video streaming forever
            reward_frame = 0
            # the action is from the last decision
            # this is to make the framework similar to the real
            # 1.time            : physical time
            # 2.time_interval   : time duration in this step
            # 3.send_data_size  : download frame data size in this step
            # 4.chunk_len       : frame time len
            # 5.rebuf           : rebuf time in this step
            # 6.buffer_size     : current client buffer_size in this step
            # 7.play_time_len   : played time len  in this step
            # 8.end_delay       : end to end latency which means the (upload end timestamp - play end timestamp)
            # 9.cdn_newest_id   : the newest frame id which cdn has
            # 10.download_id    : the current downloaded frame id
            # 11.cdn_has_frame  : all the frames cdn has currently
            # 12.skip_frame_time_len: ***new***
            # 13.decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't
            #                     Because the Gop is consist by I frame and P frame. Only in I frame you can skip frame
            # 14.buffer_flag    : True means the video is rebuffing , client buffer is rebuffing, doesn't play the video
            # 15.cdn_flag       : If the True cdn has no frame to get
            # 16.skip_flag      : ***new***,if skip some frames
            # 17.end_of_video   : If True, means the current video is over
            time, time_interval, send_data_size, chunk_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, \
            decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, LATENCY_LIMIT[latency_limit])

            # if not decision frame,then append the information of current frame to these record list:
            cnt += 1
            S_time_interval.append(time_interval)
            S_send_data_size.append(send_data_size)
            # S_chunk_len.append(chunk_len)
            S_buffer_size.append(buffer_size)
            S_rebuf.append(rebuf)
            S_end_delay.append(end_delay)
            S_play_time_len.append(play_time_len)
            # S_decision_flag.append(decision_flag)
            S_buffer_flag.append(buffer_flag)
            S_cdn_flag.append(cdn_flag)
            S_skip_time.append(skip_frame_time_len)

            # calculate throughput of every frame :
            cur_thr = float(S_send_data_size[-1]) / (S_time_interval[-1] + 0.00000001) / M_IN_K / M_IN_K
            if cur_thr-throughput_record[-1] > 0.0001:
                throughput_record.pop(0)
                throughput_record.append(cur_thr)

            # if S_send_data_size[-1] > 0:
            #     throughput_record.pop(0)
            #     throughput_record.append(
            #         float(S_send_data_size[-1]) / (S_time_interval[-1] + 0.00000001) / M_IN_K / M_IN_K)
            # elif S_send_data_size[-2] > 0:
            #     throughput_record.pop(0)
            #     throughput_record.append(
            #         float(S_send_data_size[-2]) / (S_time_interval[-2] + 0.00000001) / M_IN_K / M_IN_K)

            # QOE parameters setting:
            if end_delay <= 1.0:
                LANTENCY_PENALTY = 0.005
            else:
                LANTENCY_PENALTY = 0.01

            # calculate current reward of this frame:
            if not cdn_flag:
                reward_frame = frame_time_len * float(VIDEO_BIT_RATE[bit_rate]) / 1000 - REBUF_PENALTY * rebuf \
                               - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)
            reward_all += reward_frame  # the accumulated reward before next I frame

            if decision_flag:
                # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
                reward_frame_switch = -1 * SMOOTH_PENALTY * (abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / 1000)
                reward = reward_all + reward_frame_switch
                r_batch.append(reward)

                # last_bit_rate
                last_bit_rate = bit_rate
                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)
                # dequeue history record,make the earliest state info move to the state[-1]:
                state = np.roll(state, -1, axis=1)
                # next_video_chunk_sizes = [1000000, 1700000, 2400000, 3700000]
                if cdn_has_frame[0]:
                    next_video_chunk_sizes = [x[0] for x in cdn_has_frame][:4]
                    # for x in cdn_has_frame:
                    #     print(x[0])
                    # print(next_video_chunk_sizes)

                # predict throughput using recorded real throughput:
                thr_record_nozero = np.array(throughput_record).ravel()[np.flatnonzero(throughput_record)]
                discount_thr = np.zeros(len(thr_record_nozero))
                discount_thr[0] = thr_record_nozero[0]
                for i in range(1, len(thr_record_nozero)):
                    discount_thr[i] = thr_gamma * thr_record_nozero[i] + (1-thr_gamma) * discount_thr[i - 1]
                # if cnt < 10:
                #     throughput_predict = sum(throughput_record[-cnt - 1:]) / (cnt + 1)
                # else:
                #     throughput_predict = 0.15 * throughput_record[-5] + 0.15 * throughput_record[-4] + 0.15 * \
                #                          throughput_record[-3] + 0.15 * throughput_record[-2] + 0.5 * throughput_record[
                #                              -1]
                thr_variance = np.var(thr_record_nozero)
                thr_mean = np.mean(thr_record_nozero)
                # thr_max = np.max(thr_record_nozero)
                # thr_min = np.min(thr_record_nozero)

                # print(state.shape)
                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last 8 gop's quality
                bitrate_fft = np.fft.fft(state[0])
                state[1] = bitrate_fft.real
                state[2] = bitrate_fft.imag
                state[3, -1] = (buffer_size - 2.5) / 2.5  # current buffer size
                state[4, -1] = discount_thr[-1] / 5
                state[5, -1] = thr_mean / 5
                state[6, -1] = thr_variance
                state[7, -1] = end_delay  # end-to-end delay
                state[8, -1] = np.sum(S_skip_time)  # end-to-end delay
                state[9, -1] = float(sum(S_time_interval)) / 10  # 100 sec,last gop's download time
                state[10, :4] = np.array(next_video_chunk_sizes) / float(np.max(next_video_chunk_sizes)) # mega byte,next gop's size

                # added by penghuan on 20181202:add fft feature of last 8 gop's bitrate


                last_end_delay = end_delay

                # compute action probability vector
                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                bit_rate, target_buffer, latency_limit = getBitrateAndTargetbuffer(action_num) # changed on 20190509
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states
                # print(action_prob[0])

                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                # # log time_stamp, bit_rate, buffer_size, reward:
                # log_file.write(str(time) + '\t' +
                #                 str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                #                 str(TARGET_BUFFER[target_buffer]) + '\t' +
                #                 str(buffer_size) + '\t' +
                #                 str(sum(S_rebuf)) + '\t' +
                #                 str(sum(S_send_data_size)) + '\t' +
                #                 str(sum(S_end_delay)) + '\t' +
                #                 str(reward) + '\t' + str(random_trace) + '\n')
                # log_file.flush()

                # report experience to the coordinator
                if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                    exp_queue.put([s_batch[1:],  # ignore the first chuck
                                   a_batch[1:],  # since we don't have the
                                   r_batch[1:],  # control over it
                                   end_of_video,
                                   {'entropy': entropy_record}])

                    # synchronize the network parameters from the coordinator
                    actor_net_params, critic_net_params = net_params_queue.get()
                    actor.set_network_params(actor_net_params)
                    critic.set_network_params(critic_net_params)

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    del entropy_record[:]

                    # log_file.write('\n')  # so that in the log we know where video ends

                # store the state and action into batches
                if end_of_video:
                    throughput_record = [0] * thr_len
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here
                    target_buffer = DEFAULT_TARGETBUFFER
                    latency_limit = DEFAULT_LATENCY_LIMIT  # added on 20190509
                    action_num = getActionNum(bit_rate, target_buffer, latency_limit)  # changed on 20190509
                    action_vec = np.zeros(A_DIM)
                    action_vec[action_num] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)

                else:
                    action_vec = np.zeros(A_DIM)
                    action_num = getActionNum(bit_rate, target_buffer, latency_limit)  # changed on 20190509
                    action_vec[action_num] = 1
                    # action_vec[bit_rate] = 1
                    a_batch.append(action_vec)
                    s_batch.append(state)

                # set all record info to NULL to record next gop's download info:
                S_time_interval = []
                S_send_data_size = []
                # S_chunk_len = []
                S_rebuf = []
                S_buffer_size = []
                S_end_delay = []
                S_play_time_len = []
                # S_decision_flag = []
                S_buffer_flag = []
                S_cdn_flag = []
                S_skip_time = []
                # set reward_all to 0 after every bitrate change:
                reward_all = 0

            # if the current video is download-finished,and is not the I frame,we should empty all record info,
            # and set training-related record info to default value:
            if end_of_video and not decision_flag:
                # set all info to default value:
                cnt = 0
                reward_all = 0
                throughput_record = [0] * thr_len
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                target_buffer = DEFAULT_TARGETBUFFER
                latency_limit = DEFAULT_LATENCY_LIMIT   # added on 20190509

                S_time_interval = []
                S_send_data_size = []
                # S_chunk_len = []
                S_rebuf = []
                S_buffer_size = []
                S_end_delay = []
                S_play_time_len = []
                # S_decision_flag = []
                S_buffer_flag = []
                S_cdn_flag = []
                S_skip_time = []

                # set all training related record info to default value:
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                action_vec = np.zeros(A_DIM)
                action_num = getActionNum(bit_rate, target_buffer, latency_limit)  # changed on 20190509
                action_vec[action_num] = 1
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)




def main():
    np.random.seed(RANDOM_SEED)
    # assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done,main process is waiting for son process done
    coordinator.join()

# changed by penghuan on 20190509:
def getActionNum(bit_rate, target_buffer,latency_limit):
    action_num = (bit_rate * 2 + target_buffer) + latency_limit*8
    return action_num
#changed by penghuan on 20190509:
def getBitrateAndTargetbuffer(action_num):
    latency_limit = action_num // 8
    tmp = action_num % 8
    bit_rate = tmp//2
    target_buffer = tmp%2
    return int(bit_rate), int(target_buffer), int(latency_limit)


if __name__ == '__main__':
    main()
