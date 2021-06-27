import tensorflow as tf
import numpy as np
import tflearn
import time

GAMMA = 0.99
S_INFO = 8
S_LEN = 8
# A_DIM = 8
A_DIM = 8
A_BIT_DIM = 4
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
ENTROPY_EPS = 1e-6
entropy_weight = tf.placeholder(tf.float32)
VIDEO_BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # Kbps
# TARGET_BUFFER = [0.6, 1.2, 1.8, 2.4]
TARGET_BUFFER = [0.5, 1.0]
BUFFER_NORM_FACTOR = 7.0
M_IN_K = 1000.0
DEFAULT_QUALITY = 0  # default video quality without agent
DEFAULT_TARGETBUFFER = 0  # default target buffer
RAND_RANGE = 1000
record_len = 5

# NN_MODEL = "./submit/results/nn_model_ep_189000.ckpt"  # model path settings
NN_MODEL = "./results/nn_model_ep_142000.ckpt" # model path settings
# NN_MODEL = "/root/mmgc/team/team114/submit/results/nn_model.ckpt"

class Algorithm:
    def __init__(self):
        self.state = np.zeros((S_INFO, S_LEN))
        self.state = np.array(self.state)
        self.bit_rate = DEFAULT_QUALITY
        self.target_buffer = DEFAULT_TARGETBUFFER
        self.latency_limit = 2.8
        self.throughput_record = [0]*record_len
        self.call_cnt = 0
        self.IntialVars = []

    def Initial(self):
        sess = tf.Session()
        actor = ActorNetwork(sess,
                         state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
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

        self.IntialVars.append(actor)
        self.IntialVars.append(sess)
        return self.IntialVars

    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len, S_end_delay,
            S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id, download_id, cdn_has_frame, abr_init):
        actor = self.IntialVars[0]
        sess = self.IntialVars[1]

        # #####################the first bandwidth predict algorithm########################
        if S_send_data_size[-1] and S_time_interval[-1]:
            self.throughput_record.pop(0)
            self.throughput_record.append(float(S_send_data_size[-1]) / (S_time_interval[-1]) / M_IN_K / M_IN_K)
            self.call_cnt += 1
        elif S_send_data_size[-2] and S_time_interval[-2]:
            self.throughput_record.pop(0)
            self.throughput_record.append(float(S_send_data_size[-2]) / (S_time_interval[-2]) / M_IN_K / M_IN_K)
            self.call_cnt += 1

        if self.call_cnt + 1 < 6:
            throughput_predict = sum(self.throughput_record[-self.call_cnt:]) / (self.call_cnt)
        else:
            throughput_predict = 0.15 * self.throughput_record[-5] + 0.15 * self.throughput_record[-4] + 0.15 * \
                   self.throughput_record[-3] + 0.15 * self.throughput_record[-2] + 0.6 * self.throughput_record[-1]

        # if self.call_cnt + 1 > record_len:
        #     thr_var = np.var(self.throughput_record)
        #     thr_mean = np.mean(self.throughput_record)
        #     thr_median = np.median(self.throughput_record)

        # #####################the second bandwidth predict algorithm########################
        # use recorded throughput to initial these thr:
        # thr_1 = self.throughput_record[-1]
        # thr_2 = self.throughput_record[-1]
        # thr_3 = self.throughput_record[-1]
        # thr_4 = self.throughput_record[-1]
        # thr_5 = self.throughput_record[-1]
        # if S_send_data_size[-43]:
        #     thr_1 = S_send_data_size[-43] / S_time_interval[-43] / M_IN_K
        # elif S_send_data_size[-44]:
        #     thr_1 = S_send_data_size[-44] / S_time_interval[-44] / M_IN_K
        # if S_send_data_size[-31]:
        #     thr_2 = S_send_data_size[-31] / S_time_interval[-31] / M_IN_K
        # elif S_send_data_size[-32]:
        #     thr_2 = S_send_data_size[-32] / S_time_interval[-32] / M_IN_K
        # if S_send_data_size[-19]:
        #     thr_3 = S_send_data_size[-19] / S_time_interval[-19] / M_IN_K
        # elif S_send_data_size[-20]:
        #     thr_3 = S_send_data_size[-20] / S_time_interval[-20] / M_IN_K
        # if S_send_data_size[-7]:
        #     thr_4 = S_send_data_size[-7] / S_time_interval[-7] / M_IN_K
        # elif S_send_data_size[-8]:
        #     thr_4 = S_send_data_size[-8] / S_time_interval[-8] / M_IN_K
        # if S_send_data_size[-1] and S_time_interval[-1]:
        #     thr_5 = S_send_data_size[-1] / S_time_interval[-1] / M_IN_K
        # elif S_send_data_size[-2]:
        #     thr_5 = S_send_data_size[-2] / S_time_interval[-2] / M_IN_K
        # throughput_predict = (0.15 * thr_1 + 0.15 * thr_2 + 0.15 * thr_3 + 0.15 * thr_4 + 0.5 * thr_5)/ M_IN_K
        # # throughput_predict = 0.2*thr_1 + 0.225*thr_2 + 0.275*thr_3 + 0.3*thr_4
        # # throughput_predict = 0.1 * thr_1 + 0.1 * thr_2 + 0.15 * thr_3 + 0.15 * thr_4+ 0.5 * thr_5
        # self.throughput_record.pop(0)
        # self.throughput_record.append(throughput_predict)
        # ##################################################################################

        # calculate the current state info,this should be S_INFO number of terms
        # dequeue history record
        next_video_chunk_sizes = [500.0, 850.0, 1200.0, 1850.0]
        self.state = np.roll(self.state, -1, axis=1)
        self.state[0, -1] = VIDEO_BIT_RATE[self.bit_rate] / float(np.max(VIDEO_BIT_RATE))
        self.state[1, -1] = (S_buffer_size[-1] - 2.5) / 2.5  # current buffer size
        self.state[2, -1] = throughput_predict  # changed by penghuan on 20190522:/5
        self.state[3, -1] = S_rebuf[-1] / 10  # 100 sec,last gop's download time
        self.state[4, :4] = np.array(next_video_chunk_sizes) / float(np.max(next_video_chunk_sizes))
        self.state[5, -1] = (S_end_delay[-1] - S_end_delay[-2])*10
        _fft = np.fft.fft(self.state[0])
        self.state[6] = _fft.real
        self.state[7] = _fft.imag

        action_prob = actor.predict(sess, np.reshape(self.state, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        bit_rate, target_buffer = self.getBitrateAndTargetbuffer(action_num)

        # add by cs on 0107
        # if thr_var > 0.20 and thr_mean < 0.8:
        #     bit_rate = 0
        #     target_buffer = 0
        if S_buffer_flag[-1]:
            bit_rate = 0
            # target_buffer = 1 #103.655
            target_buffer = 0
        else:
            if S_buffer_size[-1] < 0.6 and throughput_predict < 1.2:
                bit_rate = 0
                target_buffer = 1

        # if S_end_delay[-1] - S_buffer_size[-1] > 1:
        #     bit_rate = 0
        #     target_buffer = 0

        self.bit_rate = bit_rate
        self.target_buffer = target_buffer
        return bit_rate, target_buffer, self.latency_limit, throughput_predict*M_IN_K

    # changed by cs on 20181127:
    def getActionNum(self, bit_rate, target_buffer):
        action_num = bit_rate * 2 + target_buffer
        return action_num

    # changed by cs on 20181127:
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

            # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')  # last 8 gop's quality
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')  # last 8 gop's buffer size
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')  # last 400 interval's throughput
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')  # last 8 gop's interval
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_BIT_DIM], 128, 4, activation='relu')  # size of next gop's I frame
            split_5 = tflearn.conv_1d(inputs[:, 5:6, :], 128, 4, activation='relu')  # last 8 gop's end-to-end delay
            split_6 = tflearn.conv_1d(inputs[:, 6:7, :], 128, 4, activation='relu')  # FFT of last 8 gop's quality:real
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :], 128, 4, activation='relu')  # FFT of last 8 gop's quality:image

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_5_flat = tflearn.flatten(split_5)
            split_6_flat = tflearn.flatten(split_6)
            split_7_flat = tflearn.flatten(split_7)

            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4_flat, split_5_flat, split_6_flat, split_7_flat], 'concat')

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

            # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')  # last 8 gop's quality
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')  # last 8 gop's buffer size
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')  # last 400 interval's throughput
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')  # last 8 gop's interval
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_BIT_DIM], 128, 4, activation='relu')  # size of next gop's I frame
            split_5 = tflearn.conv_1d(inputs[:, 5:6, :], 128, 4, activation='relu')  # last 8 gop's end-to-end delay
            split_6 = tflearn.conv_1d(inputs[:, 6:7, :], 128, 4, activation='relu')  # FFT of last 8 gop's quality:real
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :], 128, 4, activation='relu')  # FFT of last 8 gop's quality:image

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_5_flat = tflearn.flatten(split_5)
            split_6_flat = tflearn.flatten(split_6)
            split_7_flat = tflearn.flatten(split_7)

            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4_flat, split_5_flat, split_6_flat, split_7_flat], 'concat')

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
