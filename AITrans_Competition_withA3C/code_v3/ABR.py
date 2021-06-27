import tensorflow as tf
import numpy as np
import tflearn
import time

# network related params:
S_INFO = 11
S_LEN = 8
A_DIM = 8
A_BIT_DIM = 4
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
ENTROPY_EPS = 1e-6
entropy_weight = tf.placeholder(tf.float32)
# streaming related params:
VIDEO_BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # Kbps
TARGET_BUFFER = [0.5, 1.0]
DEFAULT_QUALITY = 0  # default video quality without agent
DEFAULT_TARGET_BUFFER = 0  # default target buffer
RAND_RANGE = 1000
M_IN_K = 1000.0
record_len = 10
thr_gamma = 0.7

NN_MODEL = "./results/summaryAndModel_20190509/nn_model_ep_188000.ckpt"  # model path settings 94000,85000
#NN_MODEL = "/root/mmgc/team/team114/submit/results/nn_model.ckpt"
#NN_MODEL = "./nn_model_ep_57000.ckpt"

class Algorithm:
    def __init__(self):
        self.state = np.zeros((S_INFO, S_LEN))
        # self.state = np.array(self.state)
        self.bit_rate = DEFAULT_QUALITY
        self.target_buffer = DEFAULT_TARGET_BUFFER
        self.latency_limit = 4
        self.thr_record = [0]*record_len
        self.call_cnt = 0
        # self.variance = 0
        self.initialVars = []

    def Initial(self):
        sess = tf.Session()
        actor = ActorNetwork(sess,
                             state_dim=[S_INFO, S_LEN],
                             action_dim=A_DIM,
                             learning_rate=ACTOR_LR_RATE)
        # critic = CriticNetwork(sess,
        #                     state_dim=[S_INFO, S_LEN],
        #                     learning_rate=CRITIC_LR_RATE)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        # state = np.zeros((S_INFO, S_LEN))
        # action_prob = actor.predict(sess, np.reshape(self.state, (1, S_INFO, S_LEN)))

        self.initialVars.append(actor)
        self.initialVars.append(sess)
        return self.initialVars

    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, abr_init):

        actor = self.initialVars[0]
        sess = self.initialVars[1]

        if cdn_has_frame[0]:
            next_video_frame_sizes = [x[0] for x in cdn_has_frame][:4]
        else:
            next_video_frame_sizes = [500.0, 850.0, 1200.0, 1850.0]

        # record the throughput:
        S_thr = []
        for i in reversed(range(1500)):
            if S_time_interval[-i] > 0:
                S_thr.append(S_send_data_size[-i]/S_time_interval[-i]/M_IN_K/M_IN_K)
            else:
                S_thr.append(0)
        # S_thr = [a/b for a, b in zip(S_send_data_size[-2000:],S_time_interval[-2000:])]
        S_thr_without_repeat = sorted(set(S_thr), key=S_thr.index)
        self.thr_record = S_thr_without_repeat[-record_len:]

        # calculate the  current predicted throughput,recorded throughputs' variance and mean value:
        thr_record_nozero = np.array(self.thr_record).ravel()[np.flatnonzero(self.thr_record)]
        discount_thr = np.zeros(len(thr_record_nozero))
        discount_thr[0] = thr_record_nozero[0]
        for i in range(1, len(thr_record_nozero)):
            discount_thr[i] = thr_gamma * thr_record_nozero[i] + (1 - thr_gamma) * discount_thr[i - 1]
        thr_mean = np.mean(thr_record_nozero)
        thr_variance = np.var(thr_record_nozero)

        # calculate the current state info,this should be S_INFO number of terms
        self.state = np.roll(self.state, -1, axis=1)  # dequeue history record
        self.state[0, -1] = VIDEO_BIT_RATE[self.bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last 8 gop's quality
        bitrate_fft = np.fft.fft(self.state[0])
        self.state[1] = bitrate_fft.real
        self.state[2] = bitrate_fft.imag
        self.state[3, -1] = (S_buffer_size[-1] - 2.5) / 2.5  # current buffer size
        self.state[4, -1] = discount_thr[-1] / 5
        self.state[5, -1] = thr_mean / 5
        self.state[6, -1] = thr_variance
        self.state[7, -1] = S_end_delay[-1]  # end-to-end delay
        self.state[8, -1] = np.sum(S_skip_time)  # end-to-end delay
        self.state[9, -1] = float(sum(S_time_interval)) / 10  # 100 sec,last gop's download time
        self.state[10, :4] = np.array(next_video_frame_sizes) / float(np.max(next_video_frame_sizes))  # next frame's size

        # calculate the bitrate and target buffer based on state info:
        action_prob = actor.predict(sess, np.reshape(self.state, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        bit_rate, target_buffer = self.getBitrateAndTargetbuffer(action_num)

        # add by cs on 0107
        # if self.variance > 0.20 and bit_rate_average < 0.8:
        #     bit_rate = 0
        #     target_buffer = 0
        #print(thr_mean)
        #if S_buffer_flag[-1] == True:
        #    bit_rate = 0
        #     # target_buffer = 1 #103.655
         #   target_buffer = 0
        # else:
        #    if S_buffer_size[-1] < 0.6 and throughput_predict < 1.2:
        #        bit_rate = 0
        #        target_buffer = 1
        #    elif throughput_predict > 1.2 and S_buffer_size[-1] > 0.5:
        #        bit_rate = 1
        #        # target_buffer = 0 #103.308
        #        target_buffer = 0
        # if S_end_delay[-1] - S_buffer_size[-1] > 1:
        #     bit_rate = 0
        #     target_buffer = 0

        # reset some record info when current video downloading-finished:
        if end_of_video:
            # self.if_begin = True
            self.bit_rate = DEFAULT_QUALITY
            self.target_buffer = DEFAULT_TARGET_BUFFER
            self.call_cnt = 0
            self.thr_record = [0] * record_len
            self.state = np.zeros((S_INFO, S_LEN))
        else:
            self.bit_rate = bit_rate
            self.target_buffer = target_buffer
            self.call_cnt += 1

        return bit_rate, target_buffer, self.latency_limit

    def getActionNum(self, bit_rate, target_buffer):
        action_num = bit_rate * 2 + target_buffer
        return action_num

    def getBitrateAndTargetbuffer(self, action_num):
        bit_rate = action_num // 2
        target_buffer = action_num % 2
        return int(bit_rate), int(target_buffer)


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        # changed by cs
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            reduction_indices=1, keepdims=True)),
                       -self.act_grad_weights)) \
                   + entropy_weight * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')  # last 8 gop's quality
            split_1 = tflearn.fully_connected(inputs[:, 1:2, :], 128, activation='relu')  # last 8 gop's quality's fft
            split_2 = tflearn.fully_connected(inputs[:, 2:3, :], 128, activation='relu')  # last 8 gop's quality's fft
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')  # last 8 gop's buffer size
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :], 128, 4, activation='relu')  # last 8 gop's predicted throughput
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], 128, activation='relu')  # throughput's mean
            split_6 = tflearn.fully_connected(inputs[:, 6:7, -1], 128, activation='relu')  # throughput's variance
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :], 128, 4, activation='relu')  # last 8 gop's end delay
            split_8 = tflearn.conv_1d(inputs[:, 8:9, :], 128, 4, activation='relu')  # last 8 gop's skip time
            split_9 = tflearn.conv_1d(inputs[:, 9:10, :], 128, 4, activation='relu')  # last 8 gop's time interval
            split_10 = tflearn.conv_1d(inputs[:, 10:11, :A_BIT_DIM], 128, 4,
                                       activation='relu')  # size of next gop's I frame

            split_0_flat = tflearn.flatten(split_0)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_7_flat = tflearn.flatten(split_7)
            split_8_flat = tflearn.flatten(split_8)
            split_9_flat = tflearn.flatten(split_9)
            split_10_flat = tflearn.flatten(split_10)

            merge_net = tflearn.merge(
                [split_0_flat, split_1, split_2, split_3_flat, split_4_flat, split_5, split_6,
                 split_7_flat, split_8_flat, split_9_flat, split_10_flat], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            return inputs, out

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    # def predict(self, inputs):
    #     return self.sess.run(self.out, feed_dict={
    #         self.inputs: inputs
    #     })
    # modified by penghuan on 20181123:
    def predict(self, sess, inputs):
        return sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    # def get_gradients(self, inputs, acts, act_grad_weights):
    #     return self.sess.run(self.actor_gradients, feed_dict={
    #         self.inputs: inputs,
    #         self.acts: acts,
    #         self.act_grad_weights: act_grad_weights
    #     })
    # changed by cs
    def get_gradients(self, ENTROPY_WEIGHT, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            entropy_weight: ENTROPY_WEIGHT,
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')  # last 8 gop's quality
            split_1 = tflearn.fully_connected(inputs[:, 1:2, :], 128, activation='relu')  # last 8 gop's quality's fft
            split_2 = tflearn.fully_connected(inputs[:, 2:3, :], 128, activation='relu')  # last 8 gop's quality's fft
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')  # last 8 gop's buffer size
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :], 128, 4, activation='relu')  # last 8 gop's predicted throughput
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], 128, activation='relu')  # throughput's mean
            split_6 = tflearn.fully_connected(inputs[:, 6:7, -1], 128, activation='relu')  # throughput's variance
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :], 128, 4, activation='relu')  # last 8 gop's end delay
            split_8 = tflearn.conv_1d(inputs[:, 8:9, :], 128, 4, activation='relu')  # last 8 gop's skip time
            split_9 = tflearn.conv_1d(inputs[:, 9:10, :], 128, 4, activation='relu')  # last 8 gop's time interval
            split_10 = tflearn.conv_1d(inputs[:, 10:11, :A_BIT_DIM], 128, 4,
                                       activation='relu')  # size of next gop's I frame

            split_0_flat = tflearn.flatten(split_0)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_7_flat = tflearn.flatten(split_7)
            split_8_flat = tflearn.flatten(split_8)
            split_9_flat = tflearn.flatten(split_9)
            split_10_flat = tflearn.flatten(split_10)

            merge_net = tflearn.merge(
                [split_0_flat, split_1, split_2, split_3_flat, split_4_flat, split_5, split_6,
                 split_7_flat, split_8_flat, split_9_flat, split_10_flat], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
