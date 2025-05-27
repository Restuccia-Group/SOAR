import gymnasium as gym
from sb3_contrib import QRDQN
from environment import SOAR 

all_data = '../sample.csv'
scen = 'aldrich'
thr = 10

env = SOAR(all_data, scen, thr)

policy_kwargs = dict(n_quantiles=50)
model = QRDQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log="./logs/qr-dqn_soar/",learning_rate=0.001, )
#model.learn(total_timesteps=10_000, log_interval=4)
# Train the agent and display a progress bar
model.learn(total_timesteps=30_000, tb_log_name="first_run_30_000_"+scen+"_"+str(thr),progress_bar = True)
# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
# Keep tb_log_name constant to have continuous curve (see note below)
model.learn(total_timesteps=30_000, tb_log_name="second_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="third_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="fourth_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="fifth_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="sixth_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="seventh_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="eighth_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="nineth_run_30_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)
model.learn(total_timesteps=30_000, tb_log_name="tenth_run_ 730_000_"+scen+"_"+str(thr), reset_num_timesteps=False, progress_bar= True)

model.save("qrdqn_"+scen+"_"+str(thr))

del model # remove to demonstrate saving and loading

model = QRDQN.load("qrdqn_"+scen+"_"+str(thr))

obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, _ = env.reset()
