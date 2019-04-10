import pygame
import pygame.freetype


class Keys:
    def __init__(self):
        """Stores the state of a keyboard"""
        self.up = False
        self.left = False
        self.down = False
        self.right = False
        self.shift_up = False
        self.shift_down = False


class SteeringWheel:
    def __init__(self):
        """Stores the state of a steering wheel"""
        self.steer = 0
        self.throttle = 0
        self.brake = 0
        self.shift_up = False
        self.shift_down = False


class Interface:
    def __init__(self, using_steering_wheel):
        """Interface for cummunicating with keyboards and steering wheels"""
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.font = pygame.freetype.SysFont('Ubuntu', 30)
        self.using_steering_wheel = using_steering_wheel
        if self.using_steering_wheel:
            self.steering_wheel = self.__init_steering_wheel()[0]
            self.steering_wheel_state = SteeringWheel()
        else:
            self.pressed = Keys()

    def __init_steering_wheel(self):
        """Checks for steering wheels and returns an initialized list"""
        steering_wheels = []
        for wheel_id in range(pygame.joystick.get_count()):
            steering_wheels.append(pygame.joystick.Joystick(wheel_id))
            steering_wheels[wheel_id].init()
        return steering_wheels

    def check_key(self, event_type, event_key):
        """Check a single key for a single event type"""
        for event in pygame.event.get():
            if event.type == event_type:
                if event.key == event_key:
                    return True
        return False

    def check_steering_key(self, button):
        pygame.event.pump()
        return bool(self.steering_wheel.get_button(button))

    def get_key_state(self):
        """Get states of all important keys"""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.pressed.up = True
                if event.key == pygame.K_DOWN:
                    self.pressed.down = True
                if event.key == pygame.K_LEFT:
                    self.pressed.left = True
                if event.key == pygame.K_RIGHT:
                    self.pressed.right = True
                if event.key == pygame.K_z:
                    self.pressed.shift_down = True
                if event.key == pygame.K_x:
                    self.pressed.shift_up = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.pressed.up = False
                if event.key == pygame.K_DOWN:
                    self.pressed.down = False
                if event.key == pygame.K_LEFT:
                    self.pressed.left = False
                if event.key == pygame.K_RIGHT:
                    self.pressed.right = False
                if event.key == pygame.K_z:
                    self.pressed.shift_down = False
                if event.key == pygame.K_x:
                    self.pressed.shift_up = False
        return self.pressed

    def get_steering_wheel_state(self):
        """Get the current state of the steering wheel"""
        pygame.event.pump()
        self.steering_wheel_state.steer = self.steering_wheel.get_axis(0)
        self.steering_wheel_state.throttle = self.steering_wheel.get_axis(2)
        self.steering_wheel_state.brake = self.steering_wheel.get_axis(3)
        self.steering_wheel_state.shift_up = \
            bool(self.steering_wheel.get_button(4))
        self.steering_wheel_state.shift_down = \
            bool(self.steering_wheel.get_button(5))
        return self.steering_wheel_state

    def display_act(self, act):
        """Display the current action on the interface"""
        accel = "ACCEL: " + str(act.accel)
        brake = "BRAKE: " + str(act.brake)
        gear = "GEAR: " + str(act.gear)
        steer = "STEER: " + str(act.steer)

        accel, rect_accel = self.font.render(accel, (255, 255, 255))
        brake, rect_brake = self.font.render(brake, (255, 255, 255))
        gear, rect_gear = self.font.render(gear, (255, 255, 255))
        steer, rect_steer = self.font.render(steer, (255, 255, 255))

        self.screen.blit(accel, (10, 10))
        self.screen.blit(brake, (10, rect_brake[1] + 20))
        self.screen.blit(gear, (10, 2 * rect_gear[1] + 30))
        self.screen.blit(steer, (10, 3 * rect_steer[1] + 40))
        pygame.display.flip()

    def display_background_color(self, color):
        """Fill the background with a solid color"""
        self.screen.fill(color)
        pygame.display.flip()
