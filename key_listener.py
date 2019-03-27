import pygame


class PressedKey:
    def __init__(self):
        self.up = False
        self.left = False
        self.down = False
        self.right = False
        self.shift_up = False
        self.shift_down = False


class KeyListener:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.pressed = PressedKey()

    def getKey(self):
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
