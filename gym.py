import numpy as np
import collections as col
import os
import time
import snakeoil
import action
import observation


class TorcsEnv:
    def __init__(self, manual=False):
        """API for communicating with snakeoil.py and torcs. Launches torcs"""
        print("Launching torcs...")
        if manual:
            self.__reset_torcs_manual()
        else:
            self.__reset_torcs()
        print("Connecting to torcs...")
        self.client = snakeoil.Client(p=3001)
        self.client.maxSteps = np.inf
        self.client.get_servers_input()
        self.obs = observation.Observation()
        self.obs.update_obs(self.client.S.d)
        self.act = action.Action()
        self.PI = 3.14159265359
        self.tick = 50

    def step(self, act, obs, auto_transmission=True):
        if auto_transmission:
            speed = obs.speedX
            rpm = obs.rpm
            act.gear = obs.gear

            """
            speed = obs.speedX * 300
            rpm = obs.rpm * 10000
            act.gear = int(round(((obs.gear * 7) - 1)))
            """
            if self.tick == 50:
                if act.gear == -1:
                    act.gear = 1
                elif act.gear == 0:
                    act.gear = 1
                elif act.gear == 1:
                    if rpm > 8500 or speed > 30:
                        act.gear += 1
                elif act.gear == 2:
                    if rpm > 8900 or speed > 80:
                        act.gear += 1
                    elif rpm < 4000 or speed < 30:
                        act.gear -= 1
                elif act.gear == 3:
                    if rpm > 9000 or speed > 130:
                        act.gear += 1
                    elif rpm < 6000 or speed < 80:
                        act.gear -= 1
                elif act.gear == 4:
                    if rpm > 9000 or speed > 180:
                        act.gear += 1
                    elif rpm < 7000 or speed < 130:
                        act.gear -= 1
                elif act.gear == 5:
                    if rpm > 9000 or speed > 230:
                        act.gear += 1
                    elif rpm < 7500 or speed < 180:
                        act.gear -= 1
                elif act.gear == 6:
                    if rpm < 8000 or speed < 230:
                        act.gear -= 1
            self.tick -= 1
            if self.tick == 0:
                self.tick = 50
            

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

    def reset(self, manual=False):
        """Re-launch torcs"""
        if manual:
            self.__reset_torcs_manual()
        else:
            self.__reset_torcs()
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()
        self.obs.update_obs(self.client.S.d)

    def end(self):
        """Kill torcs"""
        os.system('pkill torcs')

    def slow(self):
        """Slow down torcs"""
        os.system("sh ./scripts/slow_down.sh")

    def __reset_torcs(self):
        """Close torcs and run setup script"""
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs &')
        time.sleep(0.5)
        os.system('sh ./scripts/autostart.sh')
        time.sleep(0.5)

    def __reset_torcs_manual(self):
        """Close torcs and run manual setup script"""
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system('torcs -nodamage &')
        time.sleep(0.5)
        os.system('sh ./scripts/manualstart.sh')
        time.sleep(0.5)
