#! /usr/bin/env python3
"""Python file for running trained tflite model
"""

# ROS
import os
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
# from PIL import Image, ImageDraw, ImageFont
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
        self.img_num_det = 0
        self.img_num_raw = 0

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

    def __draw_results(self, img, boxes, scores):
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        imH, imW, _ = img.shape 
        for i in range(len(scores)):
            if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = 'Hole' # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        return img

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
      #  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Rezing image to fit network
        image_np = cv2.flip(image_np, 0)
        image_resized = cv2.resize(image_np, (self.width, self.height))

        # Saving raw image
        cv2.imwrite(f'/assets/images/raw/{self.img_num_raw:04d}.png', image_np)

        # Running inference
        boxes, classes, scores, num_detections = self.__inference(image_resized)
        
        # Saving image for detection
        if num_detections > 0:
            image = self.__draw_results(image_np, boxes, scores)
            cv2.imwrite(f'/assets/images/detection/{self.img_num_raw:04d}.png', image) # save at the same num as raw data, as it is easy to merge them together
            print(f'Found hole number: {self.img_num_det}')
            self.img_num_det += 1


        self.img_num_raw += 1


