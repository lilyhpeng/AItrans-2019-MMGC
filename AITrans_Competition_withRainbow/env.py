from collections import deque
import random
# import atari_py
import cv2
import torch
import fixed_env  # added by penghuan on 20190514
import load_trace  # added by penghuan on 20190514
import numpy as np  # added by penghuan on 20190514

S_INFO = 11
S_LEN = 8
A_DIM = 32  # 8*4=32
M_IN_K = 1000
thr_gamma = 0.7
thr_len = 5
frame_time_len = 0.04
VIDEO_BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # Kbps
LATENCY_LIMIT = [2.5, 2.8, 3.0, 3.5]
BUFFER_NORM_FACTOR = 7.0
SMOOTH_PENALTY = 0.02  #0.8
REBUF_PENALTY = 1.85  #1.5
SKIP_PENALTY = 0.5
DEFAULT_QUALITY = 0
DEFAULT_TARGET_BUFFER = 0
DEFAULT_LATENCY_LIMIT = 2



class Env():
    def __init__(self, args):
        self.device = args.device
        # self.ale = atari_py.ALEInterface()
        # self.ale.setInt('random_seed', args.seed)
        # self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        # self.ale.setInt('frame_skip', 0)
        # self.ale.setBool('color_averaging', False)
        # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        # actions = self.ale.getMinimalActionSet()
        # self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        # self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)  # use a deque(just like a list) to record history state
        self.training = True  # Consistent with model training mode
        # added by penghuan on 20190514:
        self.net_env = None
        self.state = np.zeros((S_INFO, S_LEN))
        self.last_bitrate = DEFAULT_QUALITY
        self.throughput_record = [0]*thr_len
        self.cnt = 0

        # self.S_time_interval = []
        # self.S_send_data_size = []
        # self.S_rebuf = []
        # self.S_buffer_size = []
        # self.S_end_delay = []
        # self.S_play_time_len = []
        # self.S_buffer_flag = []
        # self.S_cdn_flag = []
        # self.S_skip_time = []
        ################################

    def init_env(self):
        # added by penghuan on 20190514
        # use this function to create a net_env
        random_seed = 2
        DEBUG = False
        network_trace_dir = './dataset/network_trace/train/'
        video_trace_dir = './dataset/video_trace/AsianCup_China_Uzbekistan/frame_trace_'
        LOG_FILE_PATH = './results/AITrans_log'
        # load the trace
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
        self.net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                             all_cooked_bw=all_cooked_bw,
                                             random_seed=random_seed,
                                             logfile_path=LOG_FILE_PATH,
                                             VIDEO_SIZE_FILE=video_trace_dir,
                                             Debug=DEBUG)

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
          self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        self.cnt = 0
        self.throughput_record = [0] * thr_len
        self.last_bitrate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        return self.state
        # if self.life_termination:
        #     self.life_termination = False  # Reset flag
        #     self.ale.act(0)  # Use a no-op after loss of life
        # else:
        #     # Reset internals
        #     self._reset_buffer()
        #     self.ale.reset_game()
        #     # Perform up to 30 random no-ops before starting
        #     for _ in range(random.randrange(30)):
        #         self.ale.act(0)  # Assumes raw action 0 is always no-op
        #         if self.ale.game_over():
        #             self.ale.reset_game()
        # # Process and return "initial" state
        # observation = self._get_state()
        # self.state_buffer.append(observation)
        # self.lives = self.ale.lives()
        # return torch.stack(list(self.state_buffer), 0)

    def _getActionNum(self, bit_rate, target_buffer):
        # added by penghuan on 20190514
        # transfer bit_rate and target_buffer to action
        action_num = bit_rate * 2 + target_buffer * 1
        return action_num

    def _getBitrateAndTargetbuffer(self, action_num):
        # added by penghuan on 20190514
        # transfer action to bit_rate and target_buffer
        bit_rate = action_num // 2
        target_buffer = action_num % 2
        return int(bit_rate), int(target_buffer)

    def step(self, action):
        '''
        changed by penghuan on 20190514
        use action from agent to get next state and reward from environment
        '''
        done = False
        # initialize some gop-level info:
        reward_all = 0
        S_time_interval = []
        S_send_data_size = []
        S_rebuf = []
        S_buffer_size = []
        S_end_delay = []
        S_play_time_len = []
        S_buffer_flag = []
        S_cdn_flag = []
        S_skip_time = []
        bit_rate, target_buffer = self._getBitrateAndTargetbuffer(action)
        latency_limit = DEFAULT_LATENCY_LIMIT
        reward = 0.0
        while True:
            time, time_interval, send_data_size, chunk_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, \
            decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video = \
                self.net_env.get_video_frame(bit_rate, target_buffer, LATENCY_LIMIT[latency_limit])
            self.cnt += 1
            S_time_interval.append(time_interval)
            S_send_data_size.append(send_data_size)
            S_buffer_size.append(buffer_size)
            S_rebuf.append(rebuf)
            S_end_delay.append(end_delay)
            S_play_time_len.append(play_time_len)
            S_buffer_flag.append(buffer_flag)
            S_cdn_flag.append(cdn_flag)
            S_skip_time.append(skip_frame_time_len)

            # calculate throughput of every frame,record 10 0.5s' throughput:
            cur_thr = float(S_send_data_size[-1]) / (S_time_interval[-1] + 0.00000001) / M_IN_K / M_IN_K
            if cur_thr - self.throughput_record[-1] > 0.0001:
                self.throughput_record.pop(0)
                self.throughput_record.append(cur_thr)
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
                reward_frame_switch = -1 * SMOOTH_PENALTY * (abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[self.last_bitrate]) / 1000)
                reward = reward_all + reward_frame_switch

                self.last_bitrate = bit_rate
                # retrieve previous state
                # if self.cnt > 0:
                #     state = [np.zeros((S_INFO, S_LEN))]
                # else:
                #     state = self.last_state

                next_video_chunk_sizes = [1000000, 1700000, 2400000, 3700000]
                if cdn_has_frame[0]:
                    next_video_chunk_sizes = [x[0] for x in cdn_has_frame][:4]


                # predict throughput using recorded real throughput:
                thr_record_nozero = np.array(self.throughput_record).ravel()[np.flatnonzero(self.throughput_record)]
                discount_thr = np.zeros(len(thr_record_nozero))
                discount_thr[0] = thr_record_nozero[0]
                for i in range(1, len(thr_record_nozero)):
                    discount_thr[i] = thr_gamma * thr_record_nozero[i] + (1 - thr_gamma) * discount_thr[i - 1]

                thr_variance = np.var(thr_record_nozero)
                thr_mean = np.mean(thr_record_nozero)
                # thr_max = np.max(thr_record_nozero)
                # thr_min = np.min(thr_record_nozero)

                # print(state.shape)
                # this should be S_INFO number of terms
                # dequeue history record,make the earliest state info move to the state[-1]:
                self.state = np.roll(self.state, -1, axis=1)
                self.state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last 8 gop's quality
                bitrate_fft = np.fft.fft(self.state[0])
                self.state[1] = bitrate_fft.real
                self.state[2] = bitrate_fft.imag
                self.state[3, -1] = (buffer_size - 2.5) / 2.5  # current buffer size
                self.state[4, -1] = discount_thr[-1] / 5
                self.state[5, -1] = thr_mean / 5
                self.state[6, -1] = thr_variance
                self.state[7, -1] = end_delay  # end-to-end delay
                self.state[8, -1] = np.sum(S_skip_time)  # end-to-end delay
                self.state[9, -1] = float(sum(S_time_interval)) / 10  # 100 sec,last gop's download time
                self.state[10, :4] = np.array(next_video_chunk_sizes) / float(np.max(next_video_chunk_sizes))  # mega byte,next gop's size
                if end_of_video:
                    done = True
                break
            if end_of_video and not decision_flag:
                done = True
                break
        # self.cnt += 1
        # print(self.cnt)
        # self.state_buffer.append(torch.LongTensor(self.state))
        # self.state_buffer.append(torch.LongTensor(self.state))
        # self.state_buffer.append(torch.LongTensor(self.state))
        # self.state_buffer.append(torch.LongTensor(self.state))
        # print(len(self.state_buffer))
        state = torch.tensor(self.state, dtype=torch.float32, device=self.device)
        return state, reward, done

      # # Repeat action 4 times, max pool over last 2 frames
      # frame_buffer = torch.zeros(2, 84, 84, device=self.device)
      # reward, done = 0, False
      # for t in range(4):
      #   reward += self.ale.act(self.actions.get(action))
      #   if t == 2:
      #     frame_buffer[0] = self._get_state()
      #   elif t == 3:
      #     frame_buffer[1] = self._get_state()
      #   done = self.ale.game_over()
      #   if done:
      #     break
      # observation = frame_buffer.max(0)[0]
      # self.state_buffer.append(observation)
      # # Detect loss of life as terminal in training mode
      # if self.training:
      #   lives = self.ale.lives()
      #   if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
      #     self.life_termination = not done  # Only set flag when not truly done
      #     done = True
      #   self.lives = lives
      # # Return state, reward, done
      # return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return A_DIM  # len(self.actions),changed by penghuan on 20190517

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
