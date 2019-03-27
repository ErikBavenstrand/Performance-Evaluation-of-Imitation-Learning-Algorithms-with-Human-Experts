import numpy as np
import gym
import agent
import key_listener


def clip(v, lo, hi):
    """Makes sure that the value is between lo and hi"""
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


def get_expert_act(act, obs):
    """Get the action that the expert would preform"""
    target_speed = 100
    act.steer = obs.angle * 10 / env.PI
    act.steer -= obs.trackPos * .10
    act.steer = clip(act.steer, -1, 1)
    if obs.speedX < target_speed - (act.steer * 50):
        act.accel += .01
    else:
        act.accel -= .01
    if obs.speedX < 10:
        act.accel += 1 / (obs.speedX + .1)
    act.gear = 1
    if obs.speedX > 50:
        act.gear = 2
    if obs.speedX > 80:
        act.gear = 3
    if obs.speedX > 110:
        act.gear = 4
    if obs.speedX > 140:
        act.gear = 5
    if obs.speedX > 170:
        act.gear = 6
    act.accel = clip(act.accel, 0, 1)
    return act

# ----------------------------------------------------------------------------
# Number of sensors in observations
sensor_count = 69

# Number of availiable actions
action_count = 3

# Number of steps per iteration
steps = 4000

# Number of epochs
epoch_count = 100

# Batch size
batch_size = 32

# Episode count
episode_count = 5

# All observations and their corresponding actions are stored here
observations_all = np.zeros((0, sensor_count))
actions_all = np.zeros((0, 3))

# All observations and actions for a single iteration are stored here
observation_list = []
action_list = []

# Start torcs
env = gym.TorcsEnv(manual=True)

# Start listening to keys
keyboard = key_listener.KeyListener()

# ----------------------------------------------------------------------------
# Expert demonstration
print("#"*50)
print("Running expert algorithm...")
for i in range(steps):
    # If first iteration, get observation and action
    if i == 0:
        act = env.act
        obs = env.obs

    # Get the action from the expert
    act = get_expert_act(act, obs)

    # Add the current observation to the list of observations
    obs_list = obs.get_obs(angle=True, gear=True, opponents=True, rpm=True,
                           speedX=True, speedY=True, speedZ=True, track=True,
                           trackPos=True, wheelSpinVel=True, roll=True,
                           pitch=True, yaw=True)
    observation_list.append(obs_list)

    # Add the correspondig action to the list of actions
    act_list = act.get_act(accel=True, gear=True, steer=True)
    action_list.append(act_list)

    # Execute the action and get the new observation
    obs = env.step(act)
    key = keyboard.inputchar()
    if key:
        print(key)

# Exit torcs
env.end()

# ----------------------------------------------------------------------------
# Summarizing the demonstration
print("#"*50)
print('Packing expert data into arrays...')
for observation, action in zip(observation_list, action_list):
    # Concatenate all observations into array of arrays
    observations_all = np.concatenate([observations_all, np.reshape(
        observation, (1, sensor_count))], axis=0)

    # Concatenate all actions into array of arrays
    actions_all = np.concatenate([actions_all, np.reshape(
        action, (1, action_count))], axis=0)

# Create the learning agent
model = agent.Agent(name='model', input_num=observations_all[0].size,
                    output_num=actions_all[0].size)

# Train the model with the observations and actions availiable
model.train(observations_all, actions_all, n_epoch=epoch_count,
            batch=batch_size)

# ----------------------------------------------------------------------------
# Run the agent and aggregate new data produced by the expert
for episode in range(episode_count):
    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []

    # Restart the game for every iteration
    env = gym.TorcsEnv(manual=True)

    print("#"*50)
    print("# Episode: %d start" % episode)
    for i in range(steps):
        # If first iteration, get observation and action
        if i == 0:
            act = env.act
            obs = env.obs

        # Add the current observation to the list of observations
        obs_list = obs.get_obs(angle=True, gear=True, opponents=True, rpm=True,
                               speedX=True, speedY=True, speedZ=True,
                               track=True, trackPos=True, wheelSpinVel=True,
                               roll=True, pitch=True, yaw=True)
        observation_list.append(obs_list)

        # Let agent decide on an action and store expert action in list
        act_list = model.predict(np.reshape(obs_list, (1, 69)))
        act.set_act(act_list, accel=True, gear=True, steer=True)
        action_list.append(get_expert_act(act, obs).get_act(accel=True,
                                                            gear=True,
                                                            steer=True))

        # Execute the action and get the new observation
        obs = env.step(act)

    # Exit torcs
    env.end()

    # Summarize the observations and corresponding actions
    for observation, action in zip(observation_list, action_list):
        # Concatenate all observations into array of arrays
        observations_all = np.concatenate([observations_all, np.reshape(
            observation, (1, sensor_count))], axis=0)

        # Concatenate all actions into array of arrays
        actions_all = np.concatenate([actions_all, np.reshape(
            action, (1, action_count))], axis=0)

    # Train the model with the aggregated observations and actions
    model.train(observations_all, actions_all, n_epoch=epoch_count,
                batch=batch_size)
