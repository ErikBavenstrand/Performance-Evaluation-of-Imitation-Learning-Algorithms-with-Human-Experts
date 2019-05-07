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

# Number of steps per iteration
steps = 10000

# If track selection is done manually
manual_reset = True

# Initialize the input interface
interface = interface.Interface(True)

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

# Observations and actions for this iteration are stored here
observation_list = []
action_list = []

print("#" * 100)

# Get observation and action
act = env.act
obs = env.obs

for i in range(steps):  
    act = expert.get_expert_act(obs, flip=False)
    
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
