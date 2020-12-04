"""Python file for running trained tflite model
"""
# ROS
import rospy
from sensor_msgs.msg import Image

import cv2
import importlib.util



class tfliteInference():
    def __init__(self, model_path:str, sub_img_name:str, use_TPU:bool = False):
        self.model_path = model_path
        self.sub_img_name = sub_img_name
        self.use_TPU = use_TPU
        self.interpreter = self.__init_interpreter()

        # Get model details from interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        # Running subriber and inference
        self.subscriber = rospy.Subscriber(self.sub_img_name, Image, self.callback,  queue_size = 1)

    
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
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        boxes = interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = interpreter.get_tensor(self.output_details[3]['index'])[0]

        return boxes, classes, scores, num_detections

    def callback(self, ros_data):
        # conversion to CV2 
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        cv2.imshow('cv_img', image_np)
        cv2.waitKey(2)
  