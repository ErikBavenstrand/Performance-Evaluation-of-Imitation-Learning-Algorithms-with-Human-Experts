#!/usr/bin/python
import snakeoil

PI = 3.14159265359

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def drive(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S= c.S.d
    R= c.R.d
    target_speed=100

    # Damage Control
    target_speed-= S['damage'] * .05
    if target_speed < 25: target_speed= 25

    # Steer To Corner
    R['steer']= S['angle']*10 / PI
    # Steer To Center
    R['steer']-= S['trackPos']*.10
    R['steer']= clip(R['steer'],-1,1)

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer']*50):
        R['accel']+= .01
    else:
        R['accel']-= .01
    if S['speedX']<10:
       R['accel']+= 1/(S['speedX']+.1)

    # Traction Control System
    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
       R['accel']-= .2
    R['accel']= clip(R['accel'],0,1)

    # Automatic Transmission
    R['gear']=1
    if S['speedX']>50:
        R['gear']=2
    if S['speedX']>80:
        R['gear']=3
    if S['speedX']>110:
        R['gear']=4
    if S['speedX']>140:
        R['gear']=5
    if S['speedX']>170:
        R['gear']=6
    return

if __name__ == "__main__":
    Cs= [ snakeoil.Client(p=P) for P in [3001,3002,3003,3004] ]
    for step in range(Cs[0].maxSteps):
        for C in Cs:
            C.get_servers_input()
            snakeoil.drive_example(C)
            C.respond_to_server()
    else:
        for C in Cs: C.shutdown()