
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
episode_count = 1

# Number of steps per iteration
steps = 10000

# Number of epochs
epoch_count = 10000

# Batch size
batch_size = 16

# DAGGER decaying beta
beta_i = 0.85
decay = 0.85

# If track selection is done manually
manual_reset = True

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = False

# Learn from file demonstrations
file_demonstration = False
demonstration_folder = "./demonstrations/2,5K/"

# Model save location
model_folder = "./models/10K/DAGGER/"
model_name = "DAGGER-10000"

# Load unfinished model and training
load_unfinished = True

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
    # If model should learn from demonstrations first
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

        # Ensure that this is only run once
        file_demonstration = False
    elif load_unfinished:
        model.load(model_folder, model_name)
        print("Models loaded")
        fil = open(model_folder + "demonstration", "rb")
        observations_all = pickle.load(fil)
        print("Observations loaded")
        actions_all = pickle.load(fil)
        print("Actions loaded")
        beta_i = pickle.load(fil)
        fil.close()
        load_unfinished = False
    else:
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
                new_act_list = new_act.get_act(gas=True, steer=True)
                action_list.append(new_act_list)
                new_act.un_normalize_act()
                
                # Normalize the observation and add it to list of observations      
                obs.normalize_obs()
                obs_list = obs.get_obs(speedX=True, track=True, trackIndex=[0, 2, 4, 8, 9, 10, 14, 16, 18])
                observation_list.append(obs_list)

                # Normalize the act and add it to list of actions
                # Important to un-normalize the act before sending it to torcs
                act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
                act.set_act(act_list[0], gas=True, steer=True)
                act.un_normalize_act()

                # If expert or agent should control the car during this step
                if beta_i - random.uniform(0, 1) > 0:
                    # Execute the action and get the new observation
                    obs = env.step(new_act, obs)
                else:
                    # Execute the action and get the new observation
                    obs = env.step(act, obs)


            # Update the probability of using the expert action
            beta_i = beta_i * decay

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
    pickle.dump(beta_i, fil)
    fil.close()
