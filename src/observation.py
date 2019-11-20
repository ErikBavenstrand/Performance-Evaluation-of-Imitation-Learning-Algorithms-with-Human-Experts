import numpy as np
import defines


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

    def normalize_obs(self):
        """Normalizes the values of the observation to between 0 and 1"""
        self.angle = (self.angle + defines.PI) / (2 * defines.PI)
        self.damage = self.damage / 10000
        self.focus = [sensor / 200 for sensor in self.focus]
        self.fuel = self.fuel / 100
        self.gear = (self.gear + 1) / 7
        self.opponents = [opponent / 200 for opponent in self.opponents]
        self.rpm = self.rpm / 10000
        self.speedX = (self.speedX + 300) / 600
        self.speedY = (self.speedX + 300) / 600
        self.speedZ = (self.speedZ + 300) / 600
        self.track = [(sensor / 200) + 0.005 for sensor in self.track]
        self.trackPos = (self.trackPos + 10) / 20
        self.wheelSpinVel = [(spin + 300) / 600 for spin in self.wheelSpinVel]

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
