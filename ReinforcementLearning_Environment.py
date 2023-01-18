from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from clearml import Task, OutputModel

import gym
from gym import spaces
import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as R


def quat_to_rpy(q):
            #convert quaternion to roll, pitch, yaw
            rpy = R.from_quat(q).as_euler('xyz', degrees=True)
            #transform yaw to be between -90 and 90
            if rpy[2]>90:
                rpy[2] = rpy[2]-180
            elif rpy[2]<-90:
                rpy[2] = rpy[2]+180
            return rpy[2]


class RoboEnv(gym.Env):
    def __init__(self, RenderMode = False, Task = 'Lift'): # Add any arguments you need (Environment settings; Render mode  and task are used as examples)
        super(RoboEnv, self).__init__()
        # Initialize environment variables
        self.RenderMode = RenderMode
        self.Task = Task

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape= (8,))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(35,), dtype=np.float64)

        # Instantiate the environment
        self.env = suite.make(env_name= self.Task, 
                                robots="Panda",
                                has_renderer=self.RenderMode,
                                has_offscreen_renderer=False,
                                horizon=500,    
                                use_camera_obs=False,)


    def step(self, action):
        # Execute one time step within the environment
        # action = # Process the action if needed
        #Call the environment step function
        obs, reward, done, _ = self.env.step(action)
        # You may find it useful to create helper functions for the following

        gripper_pos = obs["robot0_eef_pos"]
        yaw_robot = quat_to_rpy(obs["robot0_eef_quat"])


        obs = np.hstack((obs["robot0_proprio-state"],self.target_pos))
        #obs = np.hstack((obs, yaw_robot))

        reward1 = 1 / np.linalg.norm(self.target_pos - gripper_pos)
        #reward2 = 1 / np.linalg.norm(self.target_yaw - yaw_robot)

        reward = reward1 
        #self.env.render()
        # done = # Calculate if the episode is done if you want to terminate the episode early
        return obs, reward, done, _

    def reset(self):
        # Reset the state of the environment to an initial state
        # Call the environment reset function
        obs = self.env.reset()
        # Reset any variables that need to be reset
        # Example of generating random values
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.8, 1.3)
        #yaw = np.random.uniform(-90, 90)

        self.target_pos = np.array([x, y, z], dtype=np.float64)
        #self.target_yaw = np.array([yaw])


        obs = np.hstack((obs["robot0_proprio-state"],self.target_pos))
        #obs = np.hstack((obs, self.target_yaw))

        obs = np.array(obs,dtype=np.float64)
        return obs

    def render(self):
        # Render the environment to the screen
    
        self.env.render()

    def close (self):
        # Close the environment
        self.env.close()

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='2022-Y2B-RoboSuite/Anouk', task_name='XYZ_model', output_uri=True)#, auto_connect_frameworks={'pytorch': False})
#output_model = OutputModel(task=task, framework="PyTorch")
#output_model.set_upload_destination(uri='http://31.204.128.128:8081')

#setting the base docker image
task.set_base_docker('deanis/robosuite:py3.8-2')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=8192)
parser.add_argument("--n_epochs", type=int, default=20)

args = parser.parse_args(args=[])

os.environ['WANDB_API_KEY'] = '4a0f4bad088fdcb4447d5381b8f4c563d98f06d3'

env = RoboEnv()

# initialize wandb project
run = wandb.init(project="reinforcementlearning_model",sync_tensorboard=True)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}", 
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            device='cuda')

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# variable for how often to save the model
time_steps = 10000
for i in range(1000):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")