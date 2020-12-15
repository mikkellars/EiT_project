#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hardware import *

driveA = Drive(5,6,0)
time.sleep(1.0)
driveB = Drive(26,13,1)
time.sleep(1.0)

driveA.forward()
driveB.forward()
driveA.setSpeed(10)
driveB.setSpeed(10)
driveA.drive()
driveB.drive()
i = 0
while i < 30:
    time.sleep(0.1)
    i += 1
driveA.brake()
driveB.brake()
time.sleep(1.0)
driveA.reverse()
driveB.reverse()
driveA.drive()
driveB.drive()
i = 0
while i < 30:
    time.sleep(0.1)
    i += 1
driveA.brake()
driveB.brake()
time.sleep(1.0)
