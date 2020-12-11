#! /usr/bin/env python3
"""Python file for running trained tflite model
"""

# ROS
import os
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image, ImageDraw, ImageFont
from gps_geotag import GeoTag
import cv2
import numpy as np
import importlib.util

class tfliteInference:
    def __init__(self, model_path:str, sub_img_name:str, use_TPU:bool = False, simulate:bool = False):
        self.use_TPU = use_TPU
        self.model_path = self.__get_model_name(model_path)
        print(self.model_path)
        self.sub_img_name = sub_img_name
        self.interpreter = self.__init_interpreter()

        # Get model details from interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        # Running subriber and inference
        # Image attributes
        self.bridge = CvBridge()
        self.img = None
        self.img_num = 0

        # GeoTag object
        gpstag = GeoTag()

        if simulate:
            print('Running on simulator')
            self.subscriber = rospy.Subscriber(self.sub_img_name, Image, self.callback_sim)
        else:
            print('Running on Frobit')
            self.subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.callback)

    def __get_model_name(self, path):
        """Gets the correct model based on use_TPU is true.
        Models must contain edgetpu in name when use_TPU is true.

        Args:
            path (str): path to models
        """
        print(self.use_TPU)
        model_path = None
        try:
            files = os.listdir(path)
        except CvBridgeError as e:
            print(e)
        
        for f in files:
            if self.use_TPU and 'edgetpu' in f:
                model_path = f'{path}/{f}'
            elif not self.use_TPU and not 'edgetpu' in f:
                model_path = f'{path}/{f}'
        
        if model_path is None:
            ValueError('Model not found. Check if name comply with name scheaming')
        
        return model_path

    def __init_interpreter(self):
        """
        """
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            interpreter = Interpreter(model_path=self.model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])     
        else:
            interpreter = Interpreter(model_path=self.model_path)

        interpreter.allocate_tensors()
        return interpreter

    def __inference(self, image):
        image = np.expand_dims(image, axis=0) 

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])[0]

        return boxes, classes, scores, num_detections

    def __draw_results(self, img, boxes, classes, scores):
        # # Run inference
        # model_start = time.time()
        # boxes, classes, scores = run_inference(interpreter, input_tensor)
        # model_end = time.time() - model_start
        # times.append(model_end)
        # Draw results on image
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        colors = {0:(255, 0, 0)}
        labels = {0:'Hole'}

        for i in range(len(boxes)):
            if scores[i] > .5:
                ymin = int(max(1, (boxes[i][0] * self.height)))
                xmin = int(max(1, (boxes[i][1] * self.width)))
                ymax = int(min(self.height, (boxes[i][2] * self.height)))
                xmax = int(min(self.width, (boxes[i][3] * self.width)))
                draw.rectangle((xmin, ymin, xmax, ymax), width=12, outline=colors[int(classes[i])])
                draw.rectangle((xmin, ymin, xmax, ymin-10), fill=colors[int(classes[i])])
                text = labels[int(classes[i])] + ' ' + str(scores[i]*100) + '%'
                draw.text((xmin+2, ymin-10), text, fill=(0,0,0), width=2, font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28))

        return image

    def callback_sim(self, Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(Image.data, "bgr8")
        except CvBridgeError as e:
            print(e)
        print('works')
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

    def callback(self, Image):
        np_arr = np.fromstring(Image.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Rezing image to fit network
        image_np = cv2.flip(image_np, 0)
        image_np = cv2.resize(image_np, (self.width, self.height))

        # Running inference
        boxes, classes, scores, num_detections = self.__inference(image_np)
        # Drawing boxes
        if num_detections > 0:
            image = self.__draw_results(image_np, boxes, classes, scores)
            # Saving image
            image.save(f'../images/img_{self.img_num:03d}.png')
            print(f'Found hole number: {self.img_num}')
            self.img_num += 1
       # cv2.imwrite(f'../images/img_{self.img_num:03d}.png', image_np)
       # self.img_num += 1
