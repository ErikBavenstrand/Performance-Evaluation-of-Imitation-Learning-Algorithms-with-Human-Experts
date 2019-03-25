#!/usr/bin/python
import numpy as np
import gym

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def get_expert_act(act, obs):
    target_speed=100
    act.steer = obs.angle * 10 / env.PI
    act.steer -= obs.trackPos * .10
    act.steer = clip(act.steer, -1, 1)
    if obs.speedX < target_speed - (act.steer * 50):
        act.accel += .01
    else:
        act.accel -= .01
    if obs.speedX < 10:
        act.accel += 1 / (obs.speedX + .1)
    if ((obs.wheelSpinVel[2] + obs.wheelSpinVel[3]) - (obs.wheelSpinVel[0] + obs.wheelSpinVel[1]) > 5):
        act.accel -= .2
    act.accel = clip(act.accel, 0, 1)
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
    return act


env = gym.TorcsEnv()

print("#"*50)
print('Running expert algorithm...')
i = 0
while True:
    if i == 0:
        act = env.act
        obs = env.obs
        i = -1
    act = get_expert_act(act, obs)
    obs = env.step(act)

env.end()