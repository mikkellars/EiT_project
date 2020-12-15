#!/bin/bash

echo 27 > /sys/class/gpio/export
sleep 1.0
echo out > /sys/class/gpio/gpio27/direction
sleep 1.0
echo 1 > /sys/class/gpio/gpio27/value
