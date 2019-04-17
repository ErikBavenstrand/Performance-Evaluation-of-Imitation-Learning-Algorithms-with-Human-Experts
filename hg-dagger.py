import pickle
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
sensor_count = 65

# Number of availiable actions
action_count = 3

# Number of DAGGER iterations
episode_count = 10

# Number of steps per DAGGER iteration
steps = 4000

# Number of epochs
epoch_count = 10000

# Batch size
batch_size = 32

# If track selection is done manually
manual_reset = True

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
                    output_num=actions_all.shape[1], ensemble_count=5)

# If the takeover button was pressed in the previous step
prev_takeover_pressed = False

# If the expert is in charge of the car
expert_in_charge = False

# List of doubt values
doubt = []

# Doubt threshold
doubt_t = 1
interface.display_background_color((0, 0, 0))

# Lists of tracks to be loaded from expert demonstrations
track_list = ["CG_Speedway_number_1-LESS"]

for track in track_list:
    infile = open("./demonstrations/" + track, "rb")
    obs_list = pickle.load(infile)
    act_list = pickle.load(infile)
    infile.close()

    # Summarize the observations and corresponding actions
    for observation, action_made in zip(obs_list, act_list):
        # Concatenate all observations into array of arrays
        observations_all = np.concatenate([observations_all, np.reshape(
            observation, (1, sensor_count))], axis=0)

        # Concatenate all actions into array of arrays
        actions_all = np.concatenate([actions_all, np.reshape(
            action_made, (1, action_count))], axis=0)

model.train(observations_all, actions_all, n_epoch=epoch_count,
            batch=batch_size)

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
    print("Doubt threshold: %f" % doubt_t)
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
        obs_list = obs.get_obs(angle=True, gear=True, opponents=True, rpm=True,
                               speedX=True, speedY=True, track=True,
                               trackPos=True, wheelSpinVel=True)

        # Normalize the act and add it to list of actions
        # Important to un-normalize the act before sending it to torcs
        act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
        mean = model.mean_of_prediction(act_list)
        cov = model.covariance_of_prediction(act_list, mean)
        act.set_act(act_list[0], gas=True, gear=True, steer=True)
        act.un_normalize_act()

        # Check if the expert wants to take over the control of the car
        take_over_pressed = interface.check_steering_key(2)
        if (take_over_pressed is True and prev_takeover_pressed is False) \
                or cov > doubt_t and expert_in_charge is False:
            expert_in_charge = not expert_in_charge
            if expert_in_charge:
                doubt.append(cov)
                expert.act.gear = act.gear
                interface.display_background_color((0, 0, 212))
            else:
                interface.display_background_color((0, 0, 0))
            prev_takeover_pressed = True
        elif take_over_pressed is False and prev_takeover_pressed is True:
            prev_takeover_pressed = False

        # If expert or agent should control the car during this step
        if expert_in_charge:
            # Get the action that the expert would take
            new_act = expert.get_expert_act(obs, display=False)
            new_act.normalize_act()
            new_act_list = new_act.get_act(gas=True,
                                           gear=True,
                                           steer=True)
            action_list.append(new_act_list)
            observation_list.append(obs_list)
            new_act.un_normalize_act()

            # Execute the action and get the new observation
            obs = env.step(new_act)
        else:
            # Execute the action and get the new observation
            obs = env.step(act)

    # Exit torcs
    env.end()

    # Summarize the observations and corresponding actions
    for observation, action_made in zip(observation_list, action_list):
        # Concatenate all observations into array of arrays
        observations_all = np.concatenate([observations_all, np.reshape(
            observation, (1, sensor_count))], axis=0)

        # Concatenate all actions into array of arrays
        actions_all = np.concatenate([actions_all, np.reshape(
            action_made, (1, action_count))], axis=0)

    # Train the model with the aggregated observations and actions
    model.train(observations_all, actions_all, n_epoch=epoch_count,
                batch=batch_size)

    doubt_t = 0
    doubt_count = len(doubt)
    for i in range(int(round(0.75 * doubt_count)), doubt_count):
        doubt_t += doubt[i]

    val_range = doubt_count - int(round(0.75 * doubt_count))
    if val_range == 0:
        doubt_t = 1
    else:
        doubt_t = doubt_t / val_range

    expert_in_charge = False
    interface.display_background_color((0, 0, 0))
