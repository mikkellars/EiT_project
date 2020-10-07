"""
This script helps with annotation of images.

Command line:

Arguments:

"""


import os
import cv2
import numpy as np


# ---------
# Arguments
# ---------

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Annotation tool')
    parser.add_argument('--data_dir', type=str, default='theme17_fence_inspection', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='', help='path to save directory')
    args = parser.parse_args()
    return args


def main(args):

    # ----------
    # Get images
    # ----------

    img_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    imgs = [cv2.imread(path, 0) for path in img_paths if path.endswith('jpg')]

    # ---------------
    # Annotate images
    # ---------------

    for img in imgs:

        while True:

            # Get user input
            s = input('Input peak value --> ')
            args.peak_val = float(s) * 100000
            print(f'Peak threshold is {args.peak_val}')

            # Filter image
            key = detect_fence(img)
            cv2.destroyAllWindows()

            # Save or terminate script
            if key == 27:
                print('Terminating because of ESC press')
                return
            elif key == 115:
                break


def detect_fence(img):

    # Convert to floating point representation and calculate
    # the Discrete Fourier transform (DFT) of the image.
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Determine coordinates to the center of the image.
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2     # center

    # Locate peaks in the FFT magnitude image.
    shifted_fft = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    # Black out the center of the FFT
    cv2.circle(shifted_fft, (ccol, crow), 10, 0, -1)

    # Threshold values
    # # I get good performance when approx 10 peaks are detected.
    # # 200000 - very fine reconstruction of the fence
    # peak_threshold = 200000 # Suitable for E_3_17
    # peak_threshold = 800000 # Suitable for E_3_16
    shifted_fft_peaks = np.greater(shifted_fft, args.peak_val) * 255.

    # Create a mask by dilating the detected peaks.
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(shifted_fft_peaks, kernel)
    mask = cv2.merge((mask[:, :], mask[:, :]))

    # Apply mask to the DFT and then return back to a
    # normal image through the inverse DFT.
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)

    # Input image.
    # cv2.imwrite("output/00_input_image.png", img)
    cv2.imshow('Input image', img)

    # The shifted fft spectrum.
    minval, maxval, a, b = cv2.minMaxLoc(shifted_fft)
    # cv2.imwrite("output/05_shifted_fft_image.png", shifted_fft * 255 / maxval)
    cv2.imshow('Shifted fft image', shifted_fft * 255 / maxval)

    # Save the fft peak mask.
    # cv2.imwrite("output/07_shifted_fft_peak_mask.png", mask[:, :, 0] * 1.)
    cv2.imshow('shifted fft peak mask', mask[:, :, 0] * 1.0)

    # Save the filtered image image.
    minval, maxval, a, b = cv2.minMaxLoc(img_back[:, :, 0])
    # cv2.imwrite("output/10_low_pass_filtered_image.png", img_back[:, :, 0] * 255. / maxval)
    cv2.imshow('10 low pass filtered image', img_back[:, :, 0] * 255. / maxval)

    # Save the filtered image multiplied with the input image.
    minval, maxval, a, b = cv2.minMaxLoc(img * img_back[:, :, 0])
    # cv2.imwrite("output/20_low_pass_filtered_image.png", img * img_back[:, :, 0] * 255 / maxval)
    cv2.imshow('20 low pass filtered image', img * img_back[:, :, 0] * 255 / maxval)

    return cv2.waitKey(0)


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    main(args)
