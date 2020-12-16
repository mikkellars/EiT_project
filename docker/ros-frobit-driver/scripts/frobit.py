#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, threading, time, collections, json, signal
from datetime import datetime
from hardware import *
from kafka import KafkaConsumer, KafkaProducer

class KafkaConsumerThread(threading.Thread):
    def __init__(self, servers, receive_callback=None):
        # Init thread
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()

        # Class variables
        self.servers = servers
        self.receive_callback = receive_callback

        # Init consumer
        self.consumer = KafkaConsumer(bootstrap_servers=self.servers,auto_offset_reset='latest',consumer_timeout_ms=1000)

        # Prepare topics
        self.subscribed_topics = []
        self.subscribe('_dims')

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            for message in self.consumer:
                event = json.loads(message.value)
                self.receive_callback(event)

            if self.stop_event.is_set():
                break

        self.consumer.close()

    def subscribe(self, topic):
        self.subscribed_topics.append(topic)
        self.consumer.subscribe(self.subscribed_topics)



class Frobit(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        
        self.servers = 'cloudbraink01'
        self.spike_consumer = KafkaConsumerThread(servers=self.servers, receive_callback=self.spikeReceived)
        self.spike_consumer.subscribe('motor_left')
        self.spike_consumer.subscribe('motor_right')
        self.spike_producer = KafkaProducer(bootstrap_servers=self.servers)
        self.left_sensor_topic = 'sensor_left'
        self.right_sensor_topic = 'sensor_right'
        self.event = dict()
        self.event['sender'] = 'frobit'
        self.event['payload'] = dict()
        self.average_over = 10
        self.spike_value = 50
        self.dts_left = collections.deque([0.0]*self.average_over, maxlen=self.average_over)
        self.dts_right = collections.deque([0.0]*self.average_over, maxlen=self.average_over)
        
        self.driveA = Drive(5,6,0)
        self.driveB = Drive(26,13,1)
        self.sensorLeft = Button(23)
        self.sensorRight = Button(24)
    
    def spikeReceived(self, event):
        if event['sender'] == 'frobit_left':
            print('spike left received')
            self.dts_left.append(self.spike_value)
        else: 
            print('spike right received')
            self.dts_right.append(self.spike_value)

    def sendSensorValues(self):
        self.event['@timestamp'] = datetime.utcnow().isoformat()
        if self.sensorLeft.getValue():
            self.event['sender'] = 'sensor_left'
            msg = json.dumps(self.event)
            spike = bytes(msg)
            self.spike_producer.send(self.left_sensor_topic, spike)
        if self.sensorRight.getValue():
            self.event['sender'] = 'sensor_right'
            msg = json.dumps(self.event)
            spike = bytes(msg)
            self.spike_producer.send(self.right_sensor_topic, spike)

    def stop(self):
        print("Stopping. Cleaning up")
        self.stop_event.set()
        self.spike_consumer.stop()
        time.sleep(1.0)
        del self.driveA
        del self.driveB
        del self.sensorLeft
        del self.sensorRight

    def run(self):
        self.spike_consumer.start()
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
        setpointB = int(sum(self.dts_left)/self.average_over)
        setpointA = int(sum(self.dts_right)/self.average_over)
        print('Left: ', setpointB, ' Right:', setpointA)
        self.driveA.setSpeed(setpointA)
        self.driveB.setSpeed(setpointB)
        self.dts_left.append(0.0)
        self.dts_right.append(0.0)
        
        self.sendSensorValues()
        
        time.sleep(0.1)

f = Frobit()
try:
    f.start()
    signal.pause()
except (KeyboardInterrupt, SystemExit):
    f.stop()




