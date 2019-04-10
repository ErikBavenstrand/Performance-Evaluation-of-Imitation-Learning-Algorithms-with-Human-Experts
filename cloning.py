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
sensor_count = 65

# Number of availiable actions
action_count = 3

# Number of demonstations that the expert preforms
expert_demonstration_count = 3

# Number of episodes the agent should run
agent_episode_count = 20

# Number of steps per expert iteration
expert_steps = 4000

# Number of steps per agent iteration
agent_steps = 4000

# Number of epochs
epoch_count = 100

# Batch size
batch_size = 32

# If track selection is done manually
manual_reset = False

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = True

# Use existing demonstrations
use_demonstrations = True
demonstrations_folder = "./demonstrations/"
demonstrations = ["CG_Speedway_number_1", "CG_track_2", "CG_track_3",
                  "Ruudskogen", "Spring"]

# All observations and their corresponding actions are stored here
observations_all = np.zeros((0, sensor_count))
actions_all = np.zeros((0, action_count))

# Initialize the input interface
interface = interface.Interface(using_steering_wheel)

# Create the expert
expert = expert.Expert(interface, automatic=automatic)

if use_demonstrations:
    for demonstration in demonstrations:
        print(demonstration)
        in_file = open(demonstrations_folder + demonstration, "rb")
        observation_list = pickle.load(in_file)
        action_list = pickle.load(in_file)
        in_file.close()
        observations_all = np.concatenate((observations_all, observation_list),
                                          axis=0)
        actions_all = np.concatenate((actions_all, action_list), axis=0)

else:
    # EXPERT DEMONSTRATION
    # ----------------------------------------------------------------------------
    print("Expert Demonstration")
    for episode in range(expert_demonstration_count):
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
            act = expert.get_expert_act(act, obs)

            # Normalize the observation and add it to list of observations
            obs.normalize_obs()
            obs_list = obs.get_obs(angle=True, gear=True, rpm=True,
                                   speedX=True, speedY=True, track=True,
                                   trackPos=True, wheelSpinVel=True)
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
        print('Packing expert data into arrays...')
        for observation, action_made in zip(observation_list, action_list):
            # Concatenate all observations into array of arrays
            observations_all = np.concatenate([observations_all, np.reshape(
                observation, (1, sensor_count))], axis=0)

            # Concatenate all actions into array of arrays
            actions_all = np.concatenate([actions_all, np.reshape(
                action_made, (1, action_count))], axis=0)

# Create the learning agent
model = agent.Agent(input_num=observations_all[0].size,
                    output_num=actions_all[0].size)

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
        obs_list = obs.get_obs(angle=True, gear=True, opponents=True, rpm=True,
                               speedX=True, speedY=True,  track=True,
                               trackPos=True, wheelSpinVel=True)

        # Normalize the act and add it to list of actions
        # Important to un-normalize the act before sending it to torcs
        act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
        act.set_act(act_list[0], gas=True, gear=True, steer=True)
        act.un_normalize_act()

        # Execute the action and get the new observation
        obs = env.step(act)

    # Exit torcs
    env.end()
