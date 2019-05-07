
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import random
import pygame
import numpy as np
import gym
import agent
import interface
import expert
import action
import defines


# DEFINES AND INITIALIZATIONS
# ----------------------------------------------------------------------------
# Number of sensors in observations
sensor_count = 10

# Number of availiable actions
action_count = 2

# Number of iterations
episode_count = 5

# Number of steps per iteration
steps = 10000

# Number of epochs
epoch_count = 10000

# Batch size
batch_size = 16

# If track selection is done manually
manual_reset = True

# Model save location
model_folder = "./models/10K/HG-DAGGER/"
model_name = "HG-DAGGER-10000"

# Initialize the input interface
interface = interface.Interface(True)

# Create the learning agent
model = agent.Agent(input_num=sensor_count,
                    output_num=action_count,
                    ensemble_count=5)

model.load(model_folder, model_name)

# Create the expert
expert = expert.Expert(interface, automatic=False)
# For benchmarking
distances = []
dist_raced = 0
track_length = 0
lap = 0
correct_direction = True

start_dist = 0
total_dist = 0
first = True

env = gym.TorcsEnv(manual=manual_reset)

print("Models loaded")
for episode in range(episode_count):
    if episode > 0:
        # Start torcs
        env = gym.TorcsEnv(manual=False)

    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []

    print("#" * 100)
    print("# Episode: %d start" % episode)

    # Get observation and action
    act = env.act
    obs = env.obs

    for i in range(steps):
        if i < 150:
            act.accel = 0.9
            act.gear = 1
            obs = env.step(act, obs, auto_transmission=False)
        else:
            # If quit key is pressed, prematurely end this run
            if interface.check_key(pygame.KEYDOWN, pygame.K_q):
                break
            
            # Normalize the observation and add it to list of observations      
            obs.normalize_obs()
            obs_list = obs.get_obs(speedX=True, track=True, trackIndex=[0, 2, 4, 8, 9, 10, 14, 16, 18])

            # Normalize the act and add it to list of actions
            # Important to un-normalize the act before sending it to torcs
            act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
            mean = model.mean_of_prediction(act_list)
            act.set_act(mean[0], gas=True, steer=True)
            act.un_normalize_act()

            
            # Execute action
            obs = env.step(act, obs)
            


            if lap == 0 and obs.distFromStart < 5:
                lap = 1

            if lap > 0:
                if obs.distFromStart < 5:
                    if (dist_raced + 1) >= track_length:
                        total_dist += track_length
                        dist_raced = 0
                if dist_raced < obs.distFromStart and (dist_raced + 5) > obs.distFromStart:
                    dist_raced = obs.distFromStart
        if obs.distFromStart > track_length:
            track_length = obs.distFromStart
            


    distances.append(dist_raced + total_dist)
    print(dist_raced + total_dist)

    # Exit torcs
    env.end()
    first = False
    dist_raced = 0
    total_dist = 0
    lap = 0

print(distances)
