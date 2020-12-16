#! /usr/bin/env python3
import rospy
from tflite_inference.tflite_inference import tfliteInference

def main():
    rospy.init_node('detection', anonymous=True)


    model_path = 'src/detection/nn_models'
    #sub_img_name = '/rrbot/camera1/image_raw' # For simulation
    sub_img_name = '/raspicam_node/image/compressed' # For frobit

    print(rospy.get_param('~use_tpu'))
  #  tfliteInference(model_path, sub_img_name, use_TPU=True, simulate=False)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()

