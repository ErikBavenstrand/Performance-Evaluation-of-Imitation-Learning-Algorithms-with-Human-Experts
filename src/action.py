import numpy as np


class Action:
    def __init__(self):
        """An action is the instructions sent to the car"""
        self.accel = 0.2
        self.brake = 0
        self.gas = 0
        self.clutch = 0
        self.gear = 1
        self.steer = 0
        self.focus = [-90, -45, 0, 45, 90]
        self.meta = 0

    def get_act(self, accel=None, brake=None, gas=None, clutch=None, gear=None,
                steer=None, focus=None, meta=None):
        """Returns the specified values in a numpy array"""
        act = np.array([])
        if accel:
            act = np.append(act, self.accel)
        if brake:
            act = np.append(act, self.brake)
        if gas:
            act = np.append(act, self.gas)
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

    def set_act(self, act, accel=None, brake=None, gas=None, clutch=None,
                gear=None, steer=None, focus=None, meta=None):
        """Set the values specified with a numpy array"""
        i = 0
        if accel:
            self.accel = act[0][i]
            i += 1
        if brake:
            self.brake = act[0][i]
            i += 1
        if gas:
            self.gas = act[0][i]
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

    def normalize_act(self):
        """Normalize action values to be between 0 and 1"""
        self.gas = (self.accel / 2) - (self.brake / 2) + 0.5
        self.gear = (self.gear + 1) / 7
        self.steer = (self.steer + 1) / 2

    def un_normalize_act(self):
        """Un-normalize action values to be betwen their original values"""
        if self.gas == 0.5:
            self.accel = 0
            self.accel = 0
        if self.gas > 0.5:
            self.accel = (self.gas - 0.5) * 2
            self.brake = 0
        elif self.gas < 0.5:
            self.brake = (0.5 - self.gas) * 2
            self.accel = 0
        self.gear = int(round((self.gear * 7) - 1))
        self.steer = (self.steer * 2) - 1

    def copy(self, act):
        """Copy an existing action to this"""
        self.accel = act.accel
        self.brake = act.brake
        self.gas = act.gas
        self.clutch = act.clutch
        self.gear = act.gear
        self.steer = act.steer
        self.focus = act.focus
        self.meta = act.meta

    def __clip(self, v, lo, hi):
        """Make sure the value v is between lo and hi"""
        if v > hi:
            return hi
        if v < lo:
            return lo
        return v
