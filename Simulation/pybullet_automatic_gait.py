"""
Simulation of SpotMicroAI and it's Kinematics 
Use a keyboard to see how it works
Use keyboard-Button to switch betweek walk on static-mode
"""
from os import system, name
import sys
sys.path.append("..")

import argparse
import matplotlib.animation as animation
import numpy as np
import time
import math
import datetime as dt
import keyboard
import random

from multiprocessing import Process
from Common.multiprocess_kb import KeyInterrupt

import kinematics as kn
from kinematicMotion import KinematicMotion, TrottingGait

rtime=time.time()

def reset():
    global rtime
    rtime=time.time()

def getRobotClass(engine):
    if engine == "mujoco":
        from spotmicroai_mujoco import Robot
    else:
        from environment import environment
        environment()
        from spotmicroai import Robot
    return Robot

def resetPose():
    # TODO: globals are bad
    global joy_x, joy_z, joy_y, joy_rz,joy_z
    joy_x, joy_y, joy_z, joy_rz = 128, 128, 128, 128

# define our clear function
def consoleClear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def main(id, command_status, engine="pybullet"):
    # Initialize robot and variables inside main function to avoid duplicate GUI
    RobotClass = getRobotClass(engine)
    robot = RobotClass(False, True, reset)

    spurWidth = robot.W/2+20
    stepLength = 0
    stepHeight = 72
    iXf = 120
    iXb = -132

    Lp = np.array([[iXf, -100, spurWidth, 1], [iXf, -100, -spurWidth, 1],
    [-50, -100, spurWidth, 1], [-50, -100, -spurWidth, 1]])

    resetPose()
    trotting = TrottingGait(robot)
    if engine == "mujoco":
        # Lower step height only for MuJoCo: its stiffer/coarser touchdown
        # dynamics make the default 60mm lift land noticeably harder than
        # in PyBullet. PyBullet keeps the original default untouched.
        robot.setUserDebugParameter(trotting.IDstepHeight, 30.0)

    s=False

    while True:
        bodyPos=robot.getPos()
        xr,yr,_,_,_=robot.getIMU()
        distance=math.sqrt(bodyPos[0]**2+bodyPos[1]**2)

        if distance>50:
            robot.resetBody()

        ir=xr/(math.pi/180)

        d=time.time()-rtime
        height = robot.getHeightParam()

        # calculate robot step command from keyboard inputs
        result_dict = command_status.get()
        print(result_dict)
        command_status.put(result_dict)

        print(robot.getAngle())

        if result_dict['StartStepping']:
            robot.feetPosition(trotting.positions(d-3, result_dict))
        else:
            robot.feetPosition(Lp)

        roll=0
        robot.bodyRotation((roll,math.pi/180*((joy_x)-128)/3,-(1/256*joy_y-0.5)))

        bodyX=50+yr*10
        robot.bodyPosition((bodyX, 40+height, -ir))

        robot.step()
        consoleClear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["pybullet", "mujoco"], default="pybullet",
                        help="Physics engine to run the simulation with")
    args = parser.parse_args()

    try:
        # Keyboard input Process
        KeyInputs = KeyInterrupt(args.engine)
        KeyProcess = Process(target=KeyInputs.keyInterrupt, args=(1, KeyInputs.key_status, KeyInputs.command_status))
        KeyProcess.start()

        # Main Process
        main(2, KeyInputs.command_status, args.engine)

        print("terminate KeyBoard Input process")
        if KeyProcess.is_alive():
            KeyProcess.terminate()

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Done... :)")