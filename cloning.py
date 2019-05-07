
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import pygame
import numpy as np
import gym
import agent
import interface
import expert
import action


# DEFINES AND INITIALIZATIONS
# ----------------------------------------------------------------------------
# Number of sensors in observations
sensor_count = 10

# Number of availiable actions
action_count = 2

# Number of iterations
episode_count = 1

# Number of steps per iteration
steps = 5000

# Number of epochs
epoch_count = 15000

# Batch size
batch_size = 16

# If track selection is done manually
manual_reset = True

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = False

# Learn from file demonstrations
file_demonstration = True
demonstration_folder = "./demonstrations/2,5K/"
demonstration_folder_record = "./demonstrations/BHC/10K/"

# Model save location
model_folder = "./models/10K/BHC/"
model_name = "BHC-10000"

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

while True:
    if file_demonstration:
        for f in os.listdir(demonstration_folder):
            # Append each track to the observation and action arrays
            print(f)
            fil = open(demonstration_folder + str(f), "rb")
            obs_list = pickle.load(fil)
            act_list = pickle.load(fil)
            fil.close()
            observations_all = np.concatenate((observations_all, obs_list),axis=0)
            actions_all = np.concatenate((actions_all, act_list), axis=0)

        # Train the model on the new data
        model.train(observations_all, actions_all, n_epoch=epoch_count, batch=batch_size)

        for f in os.listdir(demonstration_folder_record):
            # Append each track to the observation and action arrays
            print(f)
            fil = open(demonstration_folder_record + str(f), "rb")
            obs_list = pickle.load(fil)
            act_list = pickle.load(fil)
            fil.close()
            observations_all = np.concatenate((observations_all, obs_list),axis=0)
            actions_all = np.concatenate((actions_all, act_list), axis=0)
            # Train the model on the new data
            model.train(observations_all, actions_all, n_epoch=epoch_count, batch=batch_size)

        # Ensure that this is only run once
        file_demonstration = False
    else:
        # EXPERT DEMONSTRATION
        # ----------------------------------------------------------------------------
        print("Expert Demonstration")
        for episode in range(steps):
            # Start torcs
            env = gym.TorcsEnv(manual=manual_reset)

            # Reset the expert
            expert.reset_values()

            # Observations and actions for this iteration are stored here
            observation_list = []
            action_list = []

            # Expert demonstration
            print("#" * 100)
            print("# Episode: %d start" % episode)
            for i in range(expert_steps):
                # If first iteration, get observation and action
                if i == 0:
                    act = env.act
                    obs = env.obs

                # Get the action from the expert
                act = expert.get_expert_act(obs, display=False)

                # Normalize the observation and add it to list of observations
                obs.normalize_obs()
                obs_list = obs.get_obs(speedX=True, track=True, trackIndex=[0, 2, 4, 9, 14, 16, 18])

                observation_list.append(obs_list)

                # Normalize the act and add it to list of actions
                # Important to un-normalize the act before sending it to torcs
                act.normalize_act()
                act_list = act.get_act(gas=True, gear=True, steer=True)
                action_list.append(act_list)
                act.un_normalize_act()

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
    
    model.save(model_folder, model_name)
    fil = open(model_folder + "demonstration", "wb")
    pickle.dump(observations_all, fil)
    pickle.dump(actions_all, fil)
    fil.close()