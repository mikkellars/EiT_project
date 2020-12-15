#! /usr/bin/env python3
import rospy
from tflite_inference.tflite_inference import tfliteInference

def main():
    rospy.init_node('detection', anonymous=True)

    if not rospy.has_param('~use_tpu') or not rospy.has_param('~simulate'):
        ValueError('Need to set both use_tpu and simulate param')

    use_tpu = rospy.get_param('~use_tpu')
    simulate = rospy.get_param('~simulate')

    model_path = '/catkin_ws/src/detection/nn_models'

    if simulate:
        sub_img_name = '/rrbot/camera1/image_raw/compressed' # For simulation
    else:
        sub_img_name = '/raspicam_node/image/compressed' # For frobit

    tfliteInference(model_path, sub_img_name, use_TPU=use_tpu, simulate=simulate)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()

