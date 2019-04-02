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

    def get_expert_act(self, act, obs):
        """Get the action that the expert would preform"""
        new_act = action.Action()
        new_act.copy(act)

        # If expert is human or simple algorithm
        if self.automatic:
            target_speed = 100
            new_act.steer = obs.angle * 10 / defines.PI
            new_act.steer -= obs.trackPos * .10
            if obs.speedX < target_speed - (new_act.steer * 50):
                new_act.accel += .01
            else:
                new_act.accel -= .01
            if obs.speedX < 10:
                new_act.accel += 1 / (obs.speedX + .1)
            new_act.gear = 1
            if obs.speedX > 50:
                new_act.gear = 2
            if obs.speedX > 80:
                new_act.gear = 3
            if obs.speedX > 110:
                new_act.gear = 4
            if obs.speedX > 140:
                new_act.gear = 5
            if obs.speedX > 170:
                new_act.gear = 6
        else:
            # If human is using steering wheel or keyboard
            if self.interface.steering_wheel:
                steering_wheel = self.interface.get_steering_wheel_state()
                new_act.accel = (steering_wheel.throttle * -1)
                new_act.brake = (steering_wheel.brake * -2)
                new_act.steer = steering_wheel.steer * -1

                if steering_wheel.shift_up and self.prev_shift_up is False:
                    new_act.gear += 1
                    self.prev_shift_up = True
                elif steering_wheel.shift_up is False:
                    self.prev_shift_up = False

                if steering_wheel.shift_down and self.prev_shift_down is False:
                    new_act.gear -= 1
                    self.prev_shift_down = True
                elif steering_wheel.shift_down is False:
                    self.prev_shift_down = False

            else:
                key = self.interface.get_key_state()

                if key.up:
                    new_act.accel += .1
                else:
                    new_act.accel -= .1

                if key.down:
                    new_act.brake += .1
                else:
                    new_act.brake -= .1

                if key.left:
                    new_act.steer += .05
                elif key.right:
                    new_act.steer -= .05
                else:
                    new_act.steer = 0

                if key.shift_up and self.prev_shift_up is False:
                    new_act.gear += 1
                    self.prev_shift_up = True
                elif key.shift_up is False:
                    self.prev_shift_up = False

                if key.shift_down and self.prev_shift_down is False:
                    new_act.gear -= 1
                    self.prev_shift_down = True
                elif key.shift_down is False:
                    self.prev_shift_down = False

        # Make sure the values are valid
        new_act.accel = self.__clip(new_act.accel, 0, 1)
        new_act.brake = self.__clip(new_act.brake, 0, 1)
        new_act.gear = self.__clip(new_act.gear, -1, 7)
        new_act.steer = self.__clip(new_act.steer, -1, 1)
        new_act.gas = (new_act.accel / 2) - (new_act.brake / 2) + 0.5

        # Display the values on the interface
        self.interface.display_act(new_act)

        # Update act
        self.act = new_act

        return self.act
