import numpy as np
import collections as col
import os
import time
import snakeoil

class Observation:
    def __init__(self):
        self.angle = None
        self.curLapTime = None
        self.damage = None
        self.distFromStart = None
        self.distRaced = None
        self.fuel = None
        self.gear = None
        self.lastLapTime = None
        self.opponents = None
        self.racePos = None
        self.rpm = None
        self.speedX = None
        self.speedY = None
        self.speedZ = None
        self.track = None
        self.trackPos = None
        self.wheelSpinVel = None
        self.z = None
        self.focus = None
        self.x = None
        self.y = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.speedGlobalX = None
        self.speedGlobalY = None
    
    def update_obs(self, obs):
        self.angle = obs['angle']
        self.curLapTime = obs['curLapTime']
        self.damage = obs['damage']
        self.distFromStart = obs['distFromStart']
        self.distRaced = obs['distRaced']
        self.fuel = obs['fuel']
        self.gear = obs['gear']
        self.lastLapTime = obs['lastLapTime']
        self.opponents = obs['opponents']
        self.racePos = obs['racePos']
        self.rpm = obs['rpm']
        self.speedX = obs['speedX']
        self.speedY = obs['speedY']
        self.speedZ = obs['speedZ']
        self.track = obs['track']
        self.trackPos = obs['trackPos']
        self.wheelSpinVel = obs['wheelSpinVel']
        self.z = obs['z']
        self.focus = obs['focus']
        self.x = obs['x']
        self.y = obs['y']
        self.roll = obs['roll']
        self.pitch = obs['pitch']
        self.yaw = obs['yaw']
        self.speedGlobalX = obs['speedGlobalX']
        self.speedGlobalY = obs['speedGlobalY']

class Action:
    def __init__(self):
        self.accel = 0.2
        self.brake = 0
        self.clutch = 0
        self.gear = 1
        self.steer = 0
        self.focus = [-90,-45,0,45,90]
        self.meta = 0

class TorcsEnv:
    def __init__(self):
        print("Launching torcs...")
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
        print("Connecting to torcs...")
        self.client = snakeoil.Client(p=3001)
        self.client.maxSteps = np.inf
        self.client.get_servers_input()
        self.obs = Observation()
        self.obs.update_obs(self.client.S.d)
        self.act = Action()
        self.PI = 3.14159265359

    def get_obs(self):
        return self.obs

    def step(self, act):
        self.client.R.d['accel'] = act.accel
        self.client.R.d['brake'] = act.brake
        self.client.R.d['clutch'] = act.clutch
        self.client.R.d['gear'] = act.gear
        self.client.R.d['steer'] = act.steer
        self.client.R.d['focus'] = act.focus
        self.client.R.d['meta'] = act.meta

        # Apply the Agent's action into torcs
        self.client.respond_to_server()
        # Get the response of TORCS
        self.client.get_servers_input()
        # Get the current full-observation from torcs
        self.obs.update_obs(self.client.S.d)
        return self.get_obs()

    def reset(self):
        self.reset_torcs()
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil.Client(p=3001) 
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()  # Get the initial input from torcs
        self.obs.update_obs(self.client.S.d)

    def end(self):
        os.system('pkill torcs')

    def reset_torcs(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)