import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import pandas as pd
from gymnasium.spaces import Dict, Box, Discrete,MultiDiscrete

N_DISCRETE_ACTIONS=12 # [2x2, 3x3, 4x4], [dup, nodup],[80, 40]
N_CHANNELS=30
HEIGHT=1
WIDTH=1
N_DISCRETE_FEATURES=30

"""
Users: n1 - 0, n2 - 1
FPS:  [1fps, 5fps, 15fps, 30fps] => [0,1,2,3]
scenario: 0 - aldrich, 1 - bb
"""

class SOAR(gym.Env):
    """SOAR Environment that follows gym interface."""
    """
    State: Tuple of users, environment, deadline, 
    Actions: 0 = 2x2,dup,80 1 = 2x2,nodup,80 2 = 3x3,dup,80 3 = 3x3,nodup,80 4 = 4x4,dup,80 5= 4x4, nodup,80
             6 = 2x2,dup,40 7 = 2x2,nodup,40 8 = 3x3,dup,40 9 = 3x3,nodup,40 10 = 4x4,dup,40 11= 4x4, nodup,40
    Reward: antenna_config times energy spend
    
    2x2 - 20 of energy
    3x3 - 60 of energy 
    4x4 - 100 of energy
    """
    #metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, all_data, scen, thr):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = Discrete(N_DISCRETE_ACTIONS)
        # Example for using one shot vector as input (channel-first; channel-last also works):
        self.agent_scenario = scen
        self.max_time = 90
        self.all_data = pd.read_csv(all_data)

        self.losses = 0

        self.max_north = self.all_data['north'].max()
        self.min_north = self.all_data['north'].min()

        self.max_east = self.all_data['east'].max()
        self.min_east = self.all_data['east'].min()

        self.max_down = self.all_data['down'].max()
        self.min_down = self.all_data['down'].min()

        self.max_north_m = self.all_data['north_m'].max()
        self.min_north_m = self.all_data['north_m'].min()

        self.max_east_m = self.all_data['east_m'].max()
        self.min_east_m = self.all_data['east_m'].min()

        self.max_down_m = self.all_data['down_m'].max()
        self.min_down_m = self.all_data['down_m'].min()

        self.max_pitch = self.all_data['pitch'].max()
        self.min_pitch = self.all_data['pitch'].min()

        self.max_roll = self.all_data['roll'].max()
        self.min_roll = self.all_data['roll'].min()

        self.max_yaw = self.all_data['yaw'].max()
        self.min_yaw = self.all_data['yaw'].min()
        
        self.max_xgyro = self.all_data['xgyro'].max()
        self.min_xgyro = self.all_data['xgyro'].min()

        self.max_ygyro = self.all_data['ygyro'].max()
        self.min_ygyro = self.all_data['ygyro'].min()

        self.max_zgyro = self.all_data['zgyro'].max()
        self.min_zgyro = self.all_data['zgyro'].min()

        self.max_xaccel = self.all_data['xaccel'].max()
        self.min_xaccel = self.all_data['xaccel'].min()

        self.max_yaccel = self.all_data['yaccel'].max()
        self.min_yaccel = self.all_data['yaccel'].min()

        self.max_zaccel = self.all_data['zaccel'].max()
        self.min_zaccel = self.all_data['zaccel'].min()

        self.max_press_abs = self.all_data['press_abs'].max()
        self.min_press_abs = self.all_data['press_abs'].min()

        self.max_press_diff = self.all_data['press_diff'].max()
        self.min_press_diff = max(0.001,self.all_data['press_diff'].min())

        self.max_temp = self.all_data['temp'].max()
        self.min_temp = self.all_data['temp'].min()

        self.max_heading = self.all_data['heading'].max()
        self.min_heading = self.all_data['heading'].min()

        self.max_lat = self.all_data['lat'].max()
        self.min_lat = self.all_data['lat'].min()

        self.max_lon = self.all_data['lon'].max()
        self.min_lon = self.all_data['lon'].min()

        self.max_dist_1 = self.all_data['dist_1'].max()
        self.min_dist_1 = self.all_data['dist_1'].min()

        self.max_dist_2 = self.all_data['dist_2'].max()
        self.min_dist_2 = self.all_data['dist_2'].min()

        self.max_bandwidth = self.all_data['bandwidth'].max()
        self.min_bandwidth = self.all_data['bandwidth'].min()

        self.max_deadline = self.all_data['deadline'].max()
        self.min_deadline = self.all_data['deadline'].min()

        self.max_signal = self.all_data['signal'].max()
        self.min_signal = self.all_data['signal'].min()

        self.max_tx_bitrate = self.all_data['tx_bitrate'].max()
        self.min_tx_bitrate = self.all_data['tx_bitrate'].min()

        self.max_rx_bitrate = self.all_data['rx_bitrate'].max()
        self.min_rx_bitrate = self.all_data['rx_bitrate'].min()

        self.users = {0:'n1', 1:'n2'}
        self.fps = { 0:'1fps',1:'5fps',2:'15fps',3:'30fps'}
        #self.scenario = {0: 'aldrich', 1: 'bb'}  
        self.duplication = {0: 'dup', 1: 'nodup'} 
        #self.antenna = {0: '2x2', 1: '3x3', 2: '4x4'}        

        self.observation_space = Dict({"north": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "east": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "down": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "north_m": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "east_m": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "down_m": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "pitch": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "roll": Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "yaw": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "xgyro": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "ygyro": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "zgyro": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "xaccel": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "yaccel": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "zaccel": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "press_abs": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "press_diff": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "temp": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "heading": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "lat": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "lon": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "dist_1": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "dist_2": Box(low=0, high=1, shape=(1,), dtype='float64'), 
                                       "fps": Discrete(4),
                                       "users": Discrete(2),
                                       "duplication": Discrete(2),
                                       "bandwidth":Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "antenna":Discrete(3),
                                       "deadline":Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "signal":Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "tx_bitrate":Box(low=0, high=1, shape=(1,), dtype='float64'),
                                       "rx_bitrate":Box(low=0, high=1, shape=(1,), dtype='float64')
                                       }
                                       , seed=50)
        
        self.state_keys=['north','east','down','north_m','east_m','down_m','pitch',
          'roll','yaw','xgyro','ygyro','zgyro','xaccel','yaccel','zaccel',
          'press_abs','press_diff','temp','heading','lat','lon','dist_1',
          'dist_2','signal','rx_bitrate', 'tx_bitrate', 
          'users', 'fps', 'deadline', 'bandwidth', 'duplication',
          'losses', 'antenna']
        
        # Get random state from all data df
        flag = True

        while flag:
            sample = self.all_data.sample().to_dict(orient='records')[0]
            if sample['scenario'] == self.agent_scenario:
                flag = False

        self.state = dict.fromkeys(self.state_keys, None)

        self.time = sample['time']
        self.transform_sample(sample)

        self.state_number =  0
        self.threshold = thr

        #print("self.state: ", self.state)Dict(
    def normalize(self,sample, legend):
        
        if legend == 'north':
            sample = (sample - self.min_north) / (self.max_north - self.min_north)
                                                           
        if legend == 'east':
            sample = (sample - self.min_east) / (self.max_east - self.min_east)

        if legend == 'down':
            sample = (sample - self.min_down) / (self.max_down - self.min_down)

        if legend == 'north_m':
            sample = (sample - self.min_north_m) / (self.max_north_m - self.min_north_m)

        if legend == 'east_m':
            sample = (sample - self.min_east_m) / (self.max_east_m - self.min_east_m)
        
        if legend == 'down_m':
            sample = (sample - self.min_down_m) / (self.max_down_m - self.min_down_m)

        if legend == 'pitch':
            sample = (sample - self.min_pitch) / (self.max_pitch - self.min_pitch)

        if legend == 'yaw':
            sample = (sample - self.min_yaw) / (self.max_yaw - self.min_yaw)

        if legend == 'roll':
            sample = (sample - self.min_roll) / (self.max_roll - self.min_roll)

        if legend == 'xgyro':
            sample = (sample - self.min_xgyro) / (self.max_xgyro - self.min_xgyro)
        
        if legend == 'ygyro':
            sample = (sample - self.min_ygyro) / (self.max_ygyro - self.min_ygyro)

        if legend == 'zgyro':
            sample = (sample - self.min_zgyro) / (self.max_zgyro - self.min_zgyro)

        if legend == 'xaccel':
            sample = (sample - self.min_xaccel) / (self.max_xaccel - self.min_xaccel)

        if legend == 'yaccel':
            sample = (sample - self.min_yaccel) / (self.max_yaccel - self.min_yaccel)

        if legend == 'zaccel':
            sample = (sample - self.min_zaccel) / (self.max_zaccel - self.min_zaccel)
        
        if legend == 'press_abs':
            sample = (sample - self.min_press_abs) / (self.max_press_abs - self.min_press_abs)
        
        if legend == 'press_diff':
            sample = (sample - self.min_press_diff) / (self.max_press_diff - self.min_press_diff)

        if legend == 'temp':
            sample = (sample - self.min_temp) / (self.max_temp - self.min_temp)

        if legend == 'heading':
            sample = (sample - self.min_heading) / (self.max_heading - self.min_heading)

        if legend == 'lat':
            sample = (sample - self.min_lat) / (self.max_lat - self.min_lat)
        
        if legend == 'lon':
            sample = (sample - self.min_lon) / (self.max_lon - self.min_lon)
        
        if legend == 'dist_1':
            sample = (sample - self.min_dist_1) / (self.max_dist_1 - self.min_dist_1)

        if legend == 'dist_2':
            sample = (sample - self.min_dist_2) / (self.max_dist_2 - self.min_dist_2)

        if legend == 'deadline':
            sample = (sample - self.min_deadline) / (self.max_deadline - self.min_deadline)

        if legend == 'signal':
            sample = (sample - self.min_signal) / (self.max_signal - self.min_signal)

        if legend == 'tx_bitrate':
            sample = (sample - self.min_tx_bitrate) / (self.max_tx_bitrate - self.min_tx_bitrate)

        if legend == 'rx_bitrate':
            sample = (sample - self.min_rx_bitrate) / (self.max_rx_bitrate - self.min_rx_bitrate)

        return sample

    def transform_sample(self, sample):
        for element in self.state_keys:
            if element == 'fps':
                if int(sample[element][:-3])== 1:
                    self.state[element] =  0
                elif int(sample[element][:-3]) == 5:
                    self.state[element] =  1
                elif int(sample[element][:-3]) == 15:
                    self.state[element] =  2
                elif int(sample[element][:-3]) == 30:
                    self.state[element] = 3
            elif element == 'users':
                self.state[element] = int(sample[element][-1])-1
            elif element == 'losses':
                self.state[element]= sample[element]
            elif element == 'duplication':
                self.state[element]= [ 0 if sample[element] == 'dup' else  1][0]
            elif element == 'antenna':
                
                if sample[element] == '2x2':
                    self.state[element] = 0
                elif sample[element] == '3x3':
                    self.state[element] = 1
                else:
                    self.state[element] = 2
            else:
                self.state[element] = np.array([self.normalize(sample[element],element)], dtype='float64')
                #print("shape of ", element, self.state[element].shape)

    def step(self, action):
        #print("step")
        terminated = False
        truncated = False
        if self.time == self.max_time:
            terminated = True
            reward = 0

        else:
            self.time = self.time+1
            # Find a state in the next second in time 
            #0 = 2x2,dup,80 
            if action == 0:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '2x2') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
                
            #1 = 2x2,nodup,80 
            if action == 1:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '2x2') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #2 = 3x3,dup,80
            if action == 2:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '3x3') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #3 = 3x3,nodup,80 
            if action == 3:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '3x3') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #4 = 4x4,dup,80 
            if action == 4:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '4x4') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #5= 4x4, nodup,80
            if action == 5:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '4x4') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 80)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #6 = 2x2,dup,40 
            if action == 6:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '2x2') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #7 = 2x2,nodup,40 
            if action == 7:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '2x2') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #8 = 3x3,dup,40 
            if action == 8:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '3x3') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #9 = 3x3,nodup,40 
            if action == 9:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '3x3') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #10 = 4x4,dup,40 
            if action == 10:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '4x4') &
                                              (self.all_data['duplication'] == 'dup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            #11= 4x4, nodup,40
            if action == 11:
                try:
                    sample= self.all_data.loc[(self.all_data['time'] == self.time) &
                                              (self.all_data['antenna'] == '4x4') &
                                              (self.all_data['duplication'] == 'nodup') &
                                              (self.all_data['bandwidth'] == 40)].to_dict(orient='records')[0]
                except:
                    truncated=True
            
            if truncated == False:
                self.transform_sample(sample)
                reward = self.get_reward()
            else:
                reward = -1

        observation = self.state

        info ={}

        self.state_number+=1

        #print("Reward: ", reward)
        #print("Observation: ", observation)
        #print("Observation: ", len(observation))
        #print("Action: ", action)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.seed=seed
        
        flag = True
        while flag:
            sample = self.all_data.sample().to_dict(orient='records')[0]
            if sample['scenario'] == self.agent_scenario and sample['time'] == 1:
                flag = False

        self.state = dict.fromkeys(self.state_keys, None)

        self.time = sample['time']
        self.transform_sample(sample)

        self.state_number =  0
        #print("STATE: ",self.state)
        observation = self.state

        info = {}
        return observation, info

    def render(self):
        print("Render")
    
    def get_reward(self):
        if self.state['losses'] <= self.threshold:

            # Antenna
            if self.state['antenna'] == 0:
                return 1/30
            elif self.state['antenna'] == 1:
                return 1/60
            else:
                return 1/100
        else:
            return -1
