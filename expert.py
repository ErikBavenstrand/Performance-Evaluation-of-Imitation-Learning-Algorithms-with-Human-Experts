import gym
import action
import defines


class Expert:
    def __init__(self, interface, automatic=False):
        """Expert is the driver that the ANN is supposed to imitate"""
        self.prev_shift_up = False
        self.prev_shift_down = False
        self.act = action.Action()
        self.interface = interface
        self.automatic = automatic

    def __clip(self, v, lo, hi):
        """Makes sure that the value is between lo and hi"""
        if v < lo:
            return lo
        elif v > hi:
            return hi
        else:
            return v

    def get_expert_act(self, obs, display=True):
        """Get the action that the expert would preform"""
        # If expert is human or simple algorithm
        if self.automatic:
            target_speed = 100
            self.act.steer = obs.angle * 10 / defines.PI
            self.act.steer -= obs.trackPos * .10
            if obs.speedX < target_speed - (self.act.steer * 50):
                self.act.accel += .01
            else:
                self.act.accel -= .01
            if obs.speedX < 10:
                self.act.accel += 1 / (obs.speedX + .1)
            self.act.gear = 1
            if obs.speedX > 50:
                self.act.gear = 2
            if obs.speedX > 80:
                self.act.gear = 3
            if obs.speedX > 110:
                self.act.gear = 4
            if obs.speedX > 140:
                self.act.gear = 5
            if obs.speedX > 170:
                self.act.gear = 6
        else:
            # If human is using steering wheel or keyboard
            if self.interface.steering_wheel:
                steering_wheel = self.interface.get_steering_wheel_state()
                self.act.accel = (steering_wheel.throttle * -1)
                self.act.brake = (steering_wheel.brake * -1)
                self.act.steer = steering_wheel.steer * -1

                if steering_wheel.shift_up and self.prev_shift_up is False:
                    self.act.gear += 1
                    self.prev_shift_up = True
                elif steering_wheel.shift_up is False:
                    self.prev_shift_up = False

                if steering_wheel.shift_down and self.prev_shift_down is False:
                    self.act.gear -= 1
                    self.prev_shift_down = True
                elif steering_wheel.shift_down is False:
                    self.prev_shift_down = False

            else:
                key = self.interface.get_key_state()

                if key.up:
                    self.act.accel += .1
                else:
                    self.act.accel -= .1

                if key.down:
                    self.act.brake += .1
                else:
                    self.act.brake -= .1

                if key.left:
                    self.act.steer += .05
                elif key.right:
                    self.act.steer -= .05
                else:
                    self.act.steer = 0

                if key.shift_up and self.prev_shift_up is False:
                    self.act.gear += 1
                    self.prev_shift_up = True
                elif key.shift_up is False:
                    self.prev_shift_up = False

                if key.shift_down and self.prev_shift_down is False:
                    self.act.gear -= 1
                    self.prev_shift_down = True
                elif key.shift_down is False:
                    self.prev_shift_down = False

        # Make sure the values are valid
        self.act.accel = self.__clip(self.act.accel, 0, 1)
        self.act.brake = self.__clip(self.act.brake, 0, 1)
        self.act.gear = self.__clip(self.act.gear, -1, 7)
        self.act.steer = self.__clip(self.act.steer, -1, 1)
        self.act.gas = (self.act.accel / 2) - (self.act.brake / 2) + 0.5

        if display:
            # Display the values on the interface
            self.interface.display_act(self.act)

        return self.act

    def reset_values(self):
        self.act.accel = 0
        self.act.brake = 0
        self.act.gas = 0.5
        self.act.clutch = 0
        self.act.gear = 1
        self.steer = 0
        self.act.focus = [-90, -45, 0, 45, 90]
        self.act.meta = 0
