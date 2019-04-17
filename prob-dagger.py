
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
sensor_count = 29

# Number of availiable actions
action_count = 3

# Number of DAGGER iterations
episode_count = 1

# Number of episodes the agent should run
agent_episode_count = 1

# Number of steps per DAGGER iteration
steps = 5000

# Number of steps per agent iteration
agent_steps = 5000

# Number of epochs
epoch_count = 100

# Batch size
batch_size = 8

# If track selection is done manually
manual_reset = False

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = False

# All observations and their corresponding actions are stored here
observations_all = np.zeros((0, sensor_count))
actions_all = np.zeros((0, action_count))

# Initialize the input interface
interface = interface.Interface(using_steering_wheel)

# Create the expert
expert = expert.Expert(interface, automatic=automatic)

# Create the learning agent
model = agent.Agent(input_num=observations_all.shape[1],
                    output_num=actions_all.shape[1])

decay = np.log(0.0001) / (-70)
for run in range(1000):
    for episode in range(episode_count):
        # Start torcs
        env = gym.TorcsEnv(manual=manual_reset)

        # Reset the expert
        expert.reset_values()

        # Update the probability of using the expert action
        beta_i = defines.E**(-decay * run)

        # Observations and actions for this iteration are stored here
        observation_list = []
        action_list = []

        print("#" * 100)
        print("# Episode: %d start" % episode)
        print("Probalility beta: %f" % beta_i)
        for i in range(steps):
            # If first iteration, get observation and action
            if i == 0:
                act = env.act
                obs = env.obs

            # If quit key is pressed, prematurely end this run
            if interface.check_key(pygame.KEYDOWN, pygame.K_q):
                break

            # Get the action that the expert would take
            new_act = expert.get_expert_act(obs, flip=False)
            new_act.normalize_act()
            new_act_list = new_act.get_act(gas=True,
                                        gear=True,
                                        steer=True)
            action_list.append(new_act_list)
            new_act.un_normalize_act()

            # Normalize the observation and add it to list of observations
            obs.normalize_obs()
            obs_list = obs.get_obs(angle=True, gear=True, rpm=True,
                                   speedX=True, track=True, trackIndex=[0, 2, 4, 9, 14, 16, 18])
            observation_list.append(obs_list)

            # Normalize the act and add it to list of actions
            # Important to un-normalize the act before sending it to torcs
            act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
            act.set_act(act_list[0], gas=True, gear=True, steer=True)
            act.un_normalize_act()

            # If expert or agent should control the car during this step
            if beta_i - random.uniform(0, 1) > 0:
                # Execute the action and get the new observation
                obs = env.step(new_act)
            else:
                # Execute the action and get the new observation
                obs = env.step(act)

        # Exit torcs
        env.end()

        # ----------------------------------------------------------------------------
        # Summarizing the demonstration
        observations_all = np.concatenate((observations_all, observation_list),
                                        axis=0)
        actions_all = np.concatenate((actions_all, action_list), axis=0)

        # Train the model with the observations and actions availiable
        model.train(observations_all, actions_all, n_epoch=epoch_count,
                    batch=batch_size)

    # ANN RUNNING STEP
    # ----------------------------------------------------------------------------
    # Run the agent
    for episode in range(agent_episode_count):
        # Restart the game for every iteration
        env = gym.TorcsEnv(manual=manual_reset)

        print("#" * 100)
        print("# Episode: %d start" % episode)
        for i in range(agent_steps):
            # If first iteration, get observation and action
            if i == 0:
                act = env.act
                obs = env.obs

            # If quit key is pressed, prematurely end this run
            if interface.check_key(pygame.KEYDOWN, pygame.K_q):
                break

            # Normalize the observation and add it to list of observations
            obs.normalize_obs()
            obs_list = obs.get_obs(angle=True, gear=True, rpm=True,
                                   speedX=True, track=True, trackIndex=[0, 2, 4, 9, 14, 16, 18])

            # Normalize the act and add it to list of actions
            # Important to un-normalize the act before sending it to torcs
            act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
            act.set_act(act_list[0], gas=True, gear=True, steer=True)
            act.un_normalize_act()

            # Execute the action and get the new observation
            obs = env.step(act)

        # Exit torcs
        env.end()
