"""Python file for running trained tflite model
"""
import rospy
from pycoral.utils import edgetpu


class tfliteInference():
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.interpreter = edgetpu.make_interpreter(model_file)

    
    def init_interpreter(self):
        self.interpreter.allocate_tensors()