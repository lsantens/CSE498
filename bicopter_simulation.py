
import math
import time
import sim
from simple_pid import PID
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import gym
from gym import spaces
# from stable_baselines3 import PPO
from stable_baselines3 import SAC
# from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env

from gym.envs.registration import register

# register(
#     id='BicopterEnv-v0',
#     entry_point='bicopter_simulation:BicopterEnv',
# )

# env = gym.make('BicopterEnv-v0')
class robot():
         
    def __init__(self, frame_name, motor_names=[], client_id=0):  
        # If there is an existing connection
        if client_id:
                self.client_id = client_id
        else:
            self.client_id = self.open_connection()
            
        self.motors = self._get_handlers(motor_names) 
        
        # Robot frame
        self.frame =  self._get_handler(frame_name)
            
        
    def open_connection(self):
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim 
        
        if self.client_id != -1:
            print('Robot connected')
        else:
            print('Connection failed')
        return self.client_id
        
    def close_connection(self):    
        sim.simxGetPingTime(self.client_id)  # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
        sim.simxFinish(self.client_id)  # Now close the connection to CoppeliaSim:
        print('Connection closed')
    
    def isConnected(self):
        c,result = sim.simxGetPingTime(self.client_id)
        # Return true if the robot is connected
        return result > 0         
        
    def _get_handler(self, name):
        err_code, handler = sim.simxGetObjectHandle(self.client_id, name, sim.simx_opmode_blocking)
        return handler
    
    def _get_handlers(self, names):
        handlers = []
        for name in names:
            handler = self._get_handler(name)
            handlers.append(handler)
        
        return handlers

    def send_motor_velocities(self, vels):
        for motor, vel in zip(self.motors, vels):
            err_code = sim.simxSetJointTargetVelocity(self.client_id, 
                                                      motor, vel, sim.simx_opmode_streaming)      
            
    def set_position(self, position, relative_object=-1):
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)        
        sim.simxSetObjectPosition(self.client_id, self.frame, relative_object, position, sim.simx_opmode_oneshot)
        
    def simtime(self):
        return sim.simxGetLastCmdTime(self.client_id)
    
    def get_position(self, relative_object=-1):
        # Get position relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, position = sim.simxGetObjectPosition(self.client_id, self.frame, relative_object, sim.simx_opmode_blocking)        
        return np.array(position)
    
    def get_velocity(self, relative_object=-1):
        # Get velocity relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, velocity, omega = sim.simxGetObjectVelocity(self.client_id, self.frame, sim.simx_opmode_blocking)        
        return np.array(velocity), np.array(omega)
    
    def get_object_position(self, object_name):
        # Get Object position in the world frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, -1, sim.simx_opmode_blocking)
        return np.array(position)
    
    def get_velocity(self, relative_object=-1):
        # Get velocity relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, velocity, omega = sim.simxGetObjectVelocity(self.client_id, self.frame, sim.simx_opmode_blocking)        
        return np.array(velocity), np.array(omega)


    def get_object_relative_position(self, object_name):        
        # Get Object position in the robot frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, self.frame, sim.simx_opmode_blocking)
        return np.array(position)
    
    def set_float(self, f, signal='f'):
        return sim.simxSetFloatSignal(self.client_id, signal, f, sim.simx_opmode_oneshot) # removed oneshot_wait
    

    def set_servo_forces(self, servo_angle1, servo_angle2, force_motor1, force_motor2):
        self.set_float(force_motor1, 'f1')  # Force motor 1
        self.set_float(force_motor2, 'f2')  # Force motor 2
        self.set_float(servo_angle1, 't1')  # Servo 1
        self.set_float(servo_angle2, 't2')  # Servo 2

    def get_object_orientation(self, object_name):
        # Get Object position in the world frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        if err_code != 0:
            print(f"Error: Unable to get the handle for object {object_name}. Error code: {err_code}")
            return None
        res, angles = sim.simxGetObjectOrientation(self.client_id, object_h, -1, sim.simx_opmode_blocking)
        if res != 0:
            print(f"Error: Unable to get the orientation for object {object_name}. Result code: {res}")
            return None
        return np.array(angles)
    
    def circle_path(self, time_step, angular_velocity):
        radius = 2  # Adjust the radius of the circular path as needed
        x = radius * math.cos(angular_velocity*time_step)
        y = radius * math.sin(angular_velocity*time_step)
        return x, y

# Custom Gym environment for the Bicopter
class BicopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BicopterEnv, self).__init__()
        self.robot = robot('bicopterBody')
        self.action_space = gym.spaces.Box(
            low=np.array([0.75, 0.75, np.pi/4, np.pi/4], dtype=np.float32),
            high=np.array([1.1, 1.1, 3*np.pi/4, 3*np.pi/4], dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.ground_contact_duration = 0

        self.initial_position = self.robot.get_position()
        self.initial_orientation = self.robot.get_object_orientation('bicopterBody')

    def step(self, action):
        # Apply action
        f1, f2, t1, t2 = action
        self.robot.set_servo_forces(t1, t2, f1, f2)

        # Extract position, velocity, and orientation from the state
        position = self.robot.get_position()
        orientation = self.robot.get_object_orientation('bicopterBody')
        if orientation is None:
            # Handle the error, e.g., by resetting the environment
            print("Error: Unable to get orientation. Resetting environment.")
            return self.reset(), -100000000000000000, True, {"reason": "orientation_error"}

        velocity, angular_velocity = self.robot.get_velocity()      

        # Calculate reward
        reward = self._calculate_reward(position, orientation, angular_velocity, velocity)

        # Check if episode is done (define your own criteria)
        done = self._is_done(position)
            
        # Check if the bicopter is out of bounds
        if abs(position[0]) > 1.5 or abs(position[1]) > 1.5 or position[2]>2.2:  # 3x3 boundary
            # Trigger a reset if out of bounds
            return self.reset(), -100000000000000000, True, {"reason": "out_of_bounds"}
        elif velocity[1] > 10 or velocity[1] > 10 or velocity [2] > 10:
            return self.reset(), -100000000000000000, True, {"Moving too fast, must've broken sim"}

        # Combine all state information into a single array
        state = np.concatenate([position, orientation, velocity, angular_velocity])

        return state, reward, done, {}


    def reset(self):
        # Define the initial position and orientation for the bicopter
        initial_position = [0, 0, 0.01]  # Adjust as needed
        initial_orientation = [0, 0, 0]  # Adjust as needed

        # Reset the bicopter's position and orientation in CoppeliaSim
        self.robot.set_position(self.initial_position)
        # self.robot.set_orientation(self.initial_orientation)
        time.sleep(2)

        # If there are other states or parameters in CoppeliaSim to reset, do that here

        # Ensure the simulation state is consistent
        # This might involve specific API calls to CoppeliaSim, depending on your setup

        # Get the initial observation
        initial_observation = self._get_state()
        return initial_observation


    def _calculate_reward(self, position, orientation, angular_velocity, velocity):
        # Define thresholds for aggressive behavior
        max_acceptable_rotation_angle = np.radians(30)
        max_acceptable_angular_velocity = np.radians(50)

        # Calculate penalties for aggressive orientations
        orientation_penalty = np.sum(np.abs(orientation) > max_acceptable_rotation_angle) * 100  # Increased penalty scale
        angular_velocity_penalty = np.sum(np.abs(angular_velocity) > max_acceptable_angular_velocity) * 100

        # Define goal position and height reward parameters
        goal_position = np.array([0, 0, 1.95])
        distance_to_goal = np.linalg.norm(position - goal_position)
        height_reward_factor = 300.0  # Further increased height reward factor

        # Calculate height reward
        height_above_ground = position[2]  # Assuming ground level is at 0.95
        height_reward = (height_above_ground) * height_reward_factor  # Reward based on height above the ground

        # Penalize for staying close to the ground
        ground_contact_penalty = 0
        if height_above_ground < 1.0:  # Increased threshold for ground contact penalty
            ground_contact_penalty = -1000  # Increased penalty

        # Reward for upward velocity
        vertical_velocity_reward = 0
        if velocity[2] > 0:  # Positive Z velocity (upward movement)
            vertical_velocity_reward = velocity[2] * 100  # Increased reward

        # Combine these metrics into a single reward
        reward = -distance_to_goal + height_reward + ground_contact_penalty - orientation_penalty - angular_velocity_penalty + vertical_velocity_reward
        return reward

    def _is_done(self, position):
        # End episode if robot stays on the ground for too long
        if position[2] < 0.02:
            self.ground_contact_duration += 1
        else:
            self.ground_contact_duration = 0
        
        if self.ground_contact_duration > 50:  # Threshold for ending the episode
            return True

    def _get_state(self):
        # Get state information from CoppeliaSim
        position = self.robot.get_position()
        orientation = self.robot.get_object_orientation('bicopterBody')
        velocity, angular_velocity = self.robot.get_velocity()
        state = np.concatenate([position, orientation, velocity, angular_velocity])
        return state

    def render(self, mode='human', pause_duration=0.1):
            # This method can be called after each action during post-training visualization
            if mode == 'human':
                # Assuming you have a method in your robot class to visualize the state in CoppeliaSim
                self.robot.visualize_state()
                time.sleep(pause_duration)  # Pause to observe the movement
    
    def close(self):
        pass

# Initialize the environment
env = BicopterEnv()

# Create the SAC model
# model = PPO("MlpPolicy", env, verbose=1)
model = SAC("MlpPolicy", env, verbose=1)
# model = DDPG("MlpPolicy", env, verbose=1)

# Training loop
print("Starting training...")
total_episodes = 300
for episode in range(total_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # Log actions and position every N steps
        if step_count % 10 == 0:
            print(f"Step: {step_count}, Action: {action}, Position: {env.robot.get_position()}")

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

model.save("sac_bicopter")

# Testing the trained model
print("Testing the model...")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
        print(f"Test Episode {i+1} completed, Reward: {reward}")

env.close()
print("Environment closed.")




# # Post training, test the model and collect data
# positions = []
# rewards = []

# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, _ = env.step(action)
#     positions.append(env.robot.get_position())  # Assuming get_position() returns the current position
#     rewards.append(reward)
#     if done:
#         obs = env.reset()

# # Now plot the data
# times = range(len(rewards))

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(times, [pos[2] for pos in positions])  # Plot Z position over time
# plt.title('Bicopter Z Position Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Z Position')

# plt.subplot(1, 2, 2)
# plt.plot(times, rewards)
# plt.title('Rewards Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Reward')

# plt.tight_layout()
# plt.show()