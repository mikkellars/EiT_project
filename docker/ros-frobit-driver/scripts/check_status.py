#!/usr/bin/env python
# -*- coding: utf-8 -*-
from serial import Serial
import os

def parse_line_to_dict(line):
    header = [
            'Vbatt', 
            'Ibatt', 
            'motor_error', 
            'low_batt', 
            'shutdown', 
            'm1_diagnosis', 
            'm2_diagnosis'
            ]
    raw = line.split(';')[0:7]
    raw = [element.strip() for element in raw]
    raw = [int(i) for i in raw]
    raw[2] = bool(raw[2])
    raw[3] = bool(raw[3])
    raw[4] = bool(raw[4])
    raw[5] = hex(raw[5])[-1]
    raw[6] = hex(raw[6])[-1]
    return dict(zip(header, raw))



ser = Serial('/dev/ttyAMA0', 115200, timeout=1)
while True:
    try:
        line = ser.readline()
        data = parse_line_to_dict(line)
        if data['shutdown'] == 1:
            os.system("sudo poweroff")
        #print(data)
    except KeyboardInterrupt:
        print('interrupted!')
        exit(0)
    except:
        pass
