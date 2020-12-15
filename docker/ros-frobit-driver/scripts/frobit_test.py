#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, threading, time, signal
from hardware import *

class Frobit(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        
        self.driveA = Drive(5,0)
        self.driveB = Drive(6,1)
        self.sensorLeft = Button(23)
        self.sensorRight = Button(24)
   
        self.setpoint_left = 0
        self.setpoint_right = 0
        self.gain = 5
        self.max = 20

    def stop(self):
        print("Stopping. Cleaning up")
        self.stop_event.set()
        time.sleep(1.0)
        del self.driveA
        del self.driveB
        del self.sensorLeft
        del self.sensorRight

    def run(self):
        self.driveA.forward()
        self.driveB.forward()
        self.driveA.setSpeed(0)
        self.driveB.setSpeed(0)
        self.driveA.drive()
        self.driveB.drive()
        while not self.stop_event.is_set():
            self.controlLoop()
            if self.stop_event.is_set():
                break

    def controlLoop(self):
        if self.sensorLeft.getValue():
            self.setpoint_right -= self.gain
        else:
            self.setpoint_right += self.gain
            
        if self.sensorRight.getValue():
            self.setpoint_left -= self.gain
        else:
            self.setpoint_left += self.gain

        if self.setpoint_left > self.max:
            self.setpoint_left = self.max
        if self.setpoint_left < -self.max:
            self.setpoint_left = -self.max
        
        if self.setpoint_right > self.max:
            self.setpoint_right = self.max
        if self.setpoint_right < -self.max:
            self.setpoint_right = -self.max
        
        print('Left: ', self.setpoint_left, ' Right:', self.setpoint_right)
        self.driveA.setSpeed(self.setpoint_right)
        self.driveB.setSpeed(self.setpoint_left)
        
        time.sleep(0.1)

f = Frobit()
try:
    f.start()
    signal.pause()
except (KeyboardInterrupt, SystemExit):
    f.stop()




