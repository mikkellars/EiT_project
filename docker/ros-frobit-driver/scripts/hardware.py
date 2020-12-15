#!/usr/bin/env python
# -*- coding: utf-8 -*-
from interface import *
import RPi.GPIO as GPIO

class Encoder():
    def __init__(self, pinA, pinB):
        self.value = 0.0
        self.pinA = pinA
        self.pinB = pinB

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pinA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(pinB, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(self.pinA, GPIO.BOTH)
        GPIO.add_event_callback(self.pinA, self.callbackA)

        GPIO.add_event_detect(self.pinB, GPIO.BOTH)
        GPIO.add_event_callback(self.pinB, self.callbackB)

    def __del__(self):
        GPIO.remove_event_detect(self.pinA)
        GPIO.remove_event_detect(self.pinB)
        GPIO.cleanup()

    def callbackA(self, channel):
        A = GPIO.input(self.pinA)
        B = GPIO.input(self.pinB)
        if A:
            if B:
                self.value += 1
            else:
                self.value -= 1
        else:
            if B:
                self.value -= 1
            else:
                self.value += 1

    def callbackB(self, channel):
        A = GPIO.input(self.pinA)
        B = GPIO.input(self.pinB)
        if B:
            if A:
                self.value -= 1
            else:
                self.value += 1
        else:
            if A:
                self.value += 1
            else:
                self.value -= 1

    def getValue(self):
        return self.value

class Button():
    def __init__(self, pin):
        self.pin = pin
        
        exportGPIO(self.pin)
        setDirection(self.pin, 'in')
        setPullUp(self.pin)

    def __del__(self):
        unexportGPIO(self.pin)

    def getValue(self):
        return getValue(self.pin) == '0'


class Drive():
    def __init__(self, direction_pin, pwm, forward_direction ):
        self.direction = direction_pin
        self.forward_direction = forward_direction

        if self.forward_direction == 0:
            self.reverse_direction = 1
        else:
            self.reverse_direction = 0

        self.pwm = pwm
        self.direction_forward = None
        exportGPIO(self.direction)
        setDirection(self.direction,'out')
        exportPWM(self.pwm)
        setPWMperiod(pwm)
        setPWMDuty(self.pwm,0)

    def __del__(self):
        unexportPWM(self.pwm)
        unexportGPIO(self.direction)

    def brake(self):
        disablePWM(self.pwm)

    def drive(self):
        enablePWM(self.pwm)

    def forward(self):
        disablePWM(self.pwm)
        setValue(self.direction,self.forward_direction)
        enablePWM(self.pwm)
        self.direction_forward = True

    def reverse(self):
        disablePWM(self.pwm)
        setValue(self.direction,self.reverse_direction)
        enablePWM(self.pwm)
        self.direction_forward = False

    def setSpeed(self, percent):
        if percent > 0 and not self.direction_forward:
            self.forward()
        if percent < 0 and self.direction_forward:
            self.reverse()
        setPWMDuty(self.pwm,abs(percent))

