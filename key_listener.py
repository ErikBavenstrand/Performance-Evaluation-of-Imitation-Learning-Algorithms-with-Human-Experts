import sys
import termios
import atexit
from select import select


class KeyListener:
    def __init__(self):
        """Creates a keyboard object that listens for actions"""
        # save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)

        # new terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)

        atexit.register(self.set_normal_term)
        self.set_curses_term()

    def set_normal_term(self):
        """Switch back to normal terminal"""
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def set_curses_term(self):
        """Switch to unbuffered terminal"""
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def putch(self, ch):
        """Write char"""
        sys.stdout.write(ch)

    def getch(self):
        """Read char"""
        return sys.stdin.read(1)

    def getche(self):
        """Read char and write it back to terminal"""
        ch = getch()
        putch(ch)
        return ch

    def kbhit(self):
        """Listen to terminal"""
        dr, dw, de = select([sys.stdin], [], [], 0)
        return dr != []

    def inputchar(self):
        if self.kbhit():
            ch = self.getch()
            return ch
        return None
