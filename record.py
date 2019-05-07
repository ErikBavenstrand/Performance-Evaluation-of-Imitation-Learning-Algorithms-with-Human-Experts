import pickle
import pygame
import numpy as np
import gym
import interface
import expert
import action


# DEFINES AND INITIALIZATIONS
# ----------------------------------------------------------------------------
# Number of sensors in observations
sensor_count = 10

# Number of availiable actions
action_count = 2

# Number of record iterations
episode_count = 1

# Number of steps per iteration
steps = 2500

# If track selection is done manually
manual_reset = True

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = False

# file path
folder = "./demonstrations/2,5K/"
name = "Aalborg-2500"


# All observations and their corresponding actions are stored here
observations_all = np.zeros((0, sensor_count))
actions_all = np.zeros((0, action_count))

# All observations and their corresponding actions are stored here
observations_all_less = np.zeros((0, sensor_count))
actions_all_less = np.zeros((0, action_count))

# Initialize the input interface
interface = interface.Interface(using_steering_wheel)

# Create the expert
expert = expert.Expert(interface, automatic=automatic)

out_file = open(folder + name, "wb")

for episode in range(episode_count):
    # Start torcs
    env = gym.TorcsEnv(manual=manual_reset)

    # Reset the expert
    expert.reset_values()

    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []

    print("#" * 100)
    print("# Episode: %d start" % episode)
    for i in range(steps):
        # If first iteration, get observation and action
        if i == 0:
            act = env.act
            obs = env.obs

        # If quit key is pressed, prematurely end this run
        if interface.check_key(pygame.KEYDOWN, pygame.K_q):
            break

        # Normalize the observation and add it to list of observations
        obs.normalize_obs()
        obs_list = obs.get_obs(speedX=True, track=True, trackIndex=[0, 2, 4, 8, 9, 10, 14, 16, 18])

        # Get the action that the expert would take
        act = expert.get_expert_act(obs, flip=False)
        act.normalize_act()
        act_list = act.get_act(gas=True, steer=True)
        action_list.append(act_list)
        observation_list.append(obs_list)

        act.un_normalize_act()

        # Execute the action and get the new observation
        obs = env.step(act, obs)

    # Exit torcs
    env.end()

    observations_all = np.concatenate((observations_all, observation_list),
                                      axis=0)
    actions_all = np.concatenate((actions_all, action_list), axis=0)

pickle.dump(observations_all, out_file)
pickle.dump(actions_all, out_file)
out_file.close()
