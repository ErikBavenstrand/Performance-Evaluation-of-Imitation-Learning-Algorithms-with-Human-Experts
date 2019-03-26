import numpy as np
import collections as col
import os
import time
import snakeoil


class Observation:
    def __init__(self):
        """An observation is the states of all sensors on the car"""
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
        """Updates the values of the object with dictionary"""
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

    def get_obs(self, angle=None, curLapTime=None, damage=None,
                distFromStart=None, distRaced=None, fuel=None,
                gear=None, lastLapTime=None, opponents=None, racePos=None,
                rpm=None, speedX=None, speedY=None, speedZ=None, track=None,
                trackPos=None, wheelSpinVel=None, z=None, focus=None, x=None,
                y=None, roll=None, pitch=None, yaw=None, speedGlobalX=None,
                speedGlobalY=None):
        """Return the specified values in a numpy array"""
        obs = np.array([])
        if angle:
            obs = np.append(obs, self.angle)
        if curLapTime:
            obs = np.append(obs, self.curLapTime)
        if damage:
            obs = np.append(obs, self.damage)
        if distFromStart:
            obs = np.append(obs, self.distFromStart)
        if distRaced:
            obs = np.append(obs, self.distRaced)
        if fuel:
            obs = np.append(obs, self.fuel)
        if gear:
            obs = np.append(obs, self.gear)
        if lastLapTime:
            obs = np.append(obs, self.lastLapTime)
        if opponents:
            obs = np.append(obs, self.opponents)
        if racePos:
            obs = np.append(obs, self.racePos)
        if rpm:
            obs = np.append(obs, self.rpm)
        if speedX:
            obs = np.append(obs, self.speedX)
        if speedY:
            obs = np.append(obs, self.speedY)
        if speedZ:
            obs = np.append(obs, self.speedZ)
        if track:
            obs = np.append(obs, self.track)
        if trackPos:
            obs = np.append(obs, self.trackPos)
        if wheelSpinVel:
            obs = np.append(obs, self.wheelSpinVel)
        if z:
            obs = np.append(obs, self.z)
        if focus:
            obs = np.append(obs, self.focus)
        if x:
            obs = np.append(obs, self.x)
        if y:
            obs = np.append(obs, self.y)
        if roll:
            obs = np.append(obs, self.roll)
        if pitch:
            obs = np.append(obs, self.pitch)
        if yaw:
            obs = np.append(obs, self.yaw)
        if speedGlobalX:
            obs = np.append(obs, self.speedGlobalX)
        if speedGlobalY:
            obs = np.append(obs, self.speedGlobalY)
        return obs


class Action:
    def __init__(self):
        """An action is the instructions sent to the car"""
        self.accel = 0.2
        self.brake = 0
        self.clutch = 0
        self.gear = 1
        self.steer = 0
        self.focus = [-90, -45, 0, 45, 90]
        self.meta = 0

    def get_act(self, accel=None, brake=None, clutch=None, gear=None,
                steer=None, focus=None, meta=None):
        """Returns the specified values in a numpy array"""
        act = np.array([])
        if accel:
            act = np.append(act, self.accel)
        if brake:
            act = np.append(act, self.brake)
        if clutch:
            act = np.append(act, self.clutch)
        if gear:
            act = np.append(act, self.gear)
        if steer:
            act = np.append(act, self.steer)
        if focus:
            act = np.append(act, self.focus)
        if meta:
            act = np.append(act, self.meta)
        return act

    def set_act(self, act, accel=None, brake=None, clutch=None, gear=None,
                steer=None, focus=None, meta=None):
        """Set the values specified with a numpy array"""
        i = 0
        if accel:
            self.accel = act[0][i]
            i += 1
        if brake:
            self.brake = act[0][i]
            i += 1
        if clutch:
            self.clutch = act[0][i]
            i += 1
        if gear:
            self.gear = act[0][i]
            i += 1
        if steer:
            self.steer = act[0][i]
            i += 1
        if focus:
            self.focus = act[0][i]
            i += 1
        if meta:
            self.meta = act[0][i]
            i += 1


class TorcsEnv:
    def __init__(self):
        """API for communicating with snakeoil.py and torcs. Launches torcs"""
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

    def step(self, act):
        """Preforms a single action and returns the reply by torcs"""
        self.client.R.d['accel'] = act.accel
        self.client.R.d['brake'] = act.brake
        self.client.R.d['clutch'] = act.clutch
        self.client.R.d['gear'] = act.gear
        self.client.R.d['steer'] = act.steer
        self.client.R.d['focus'] = act.focus
        self.client.R.d['meta'] = act.meta
        self.client.respond_to_server()
        self.client.get_servers_input()
        self.obs.update_obs(self.client.S.d)
        return self.obs

    def reset(self):
        """Re-launch torcs"""
        self.reset_torcs()
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()
        self.obs.update_obs(self.client.S.d)

    def end(self):
        """Kill torcs"""
        os.system('pkill torcs')

    def reset_torcs(self):
        """Close torcs and run setup script"""
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
