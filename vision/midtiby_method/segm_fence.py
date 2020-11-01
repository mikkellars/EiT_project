"""
Segmentation of fence, developed by Henrik Skov Midtiby.
"""


import os
import sys
sys.path.append(os.getcwd())

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation of fence')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/test_set', help='')
    args = parser.parse_args()
    return args


def main(args):
    im_path = f'{args.data_dir}/images'
    images = [f'{im_path}/{f}' for f in os.listdir(im_path)]

    label_path = f'{args.data_dir}/labels'
    labels = [f'{label_path}/{f}' for f in os.listdir(label_path)]

    for image, label in zip(images, labels):
        image = load_image(image)

        filtered_im, filtered_im_w_input = segment_fence(image)

        # Sobel
        grad_x = cv2.Sobel(filtered_im_w_input, cv2.CV_16S, 1, 0, ksize=3, scale=2, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(filtered_im_w_input, cv2.CV_16S, 0, 1, ksize=3, scale=2, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        plt.imshow(dst, cmap='gray')
        plt.title('')
        plt.axis('off')
        plt.show()

        # Show ground truth
        label = load_image(label)
        plt.imshow(label, cmap='gray')
        plt.title('Grund truth label')
        plt.axis('off')
        plt.show()


def load_image(path:str, mode='L')->np.array:
    im = Image.open(path).convert(mode)
    im = np.array(im)
    im = cv2.resize(im, (400, 400))
    return im


def segment_fence(im:np.array, n_peaks:int=10, show:bool=True)->np.array:

    # Convert to floatingpoint representation and calculate the DFT of the image
    im_fp32 = np.float32(im)
    dft = cv2.dft(im_fp32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Determine coordinates to the center of the image
    rows, cols = im.shape
    c_row, c_col = rows // 2, cols // 2

    # Locate peaks in the FFT magnitue image
    shifted_fft = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Black out the center of the FFT
    cv2.circle(shifted_fft, (c_col, c_row), 10, 0, -1)

    # Threshold values, good performance when 10 peaks are detected.
    # 200000 - 800000 (E_3_17: 200000 and E_3_16: 800000)
    for peak_threshold in range(100000, 800000, 5000):
        shifted_fft_peaks = np.greater(shifted_fft, peak_threshold) * 255.0
        peaks = np.count_nonzero(shifted_fft_peaks)
        if peaks == n_peaks: break

    if peaks != n_peaks:
        print(f'Could only find {peaks} peaks.')
        return None

    # Create a mask by dilating the detected peaks
    ksize = 11
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.dilate(shifted_fft_peaks, kernel)
    mask = cv2.merge((mask[:, :], mask[:, :]))

    # Apply the mask to the DFT and then return back to a normal image through the inverse DFT
    fshift = dft_shift * mask
    f_inv_shift = np.fft.ifftshift(fshift)
    im_back = cv2.idft(f_inv_shift)

    # Shifted FFT spectrum
    min_val, max_val, a, b = cv2.minMaxLoc(shifted_fft)
    shifted_fft_spectrum = shifted_fft * 255 / max_val

    # FFT peak mask
    fft_peak_mask = mask[:, :, 0] * 1.0

    # Filtered image
    min_val, max_val, a, b = cv2.minMaxLoc(im_back[:, :, 0])
    filtered_im = im_back[:, :, 0] * 255.0 / max_val

    # Filtered image multiplied with the input image
    min_val, max_val, a, b = cv2.minMaxLoc(im * im_back[:, :, 0])
    filtered_im_w_input = im * im_back[:, :, 0] * 255.0 / max_val

    if show:
        # # Show input image
        # plt.imshow(im, cmap='gray')
        # plt.title('Input image')
        # plt.axis('off')
        # plt.show()

        # # Show the shifted fft spectrum
        # plt.imshow(shifted_fft_spectrum, cmap='gray')
        # plt.title('Shifted fft image')
        # plt.axis('off')
        # plt.show()

        # # Show the fft peak mask
        # plt.imshow(fft_peak_mask, cmap='gray')
        # plt.title('Shifted fft peak mask')
        # plt.axis('off')
        # plt.show()

        # Show the filtered image
        plt.imshow(filtered_im, cmap='gray')
        plt.title('Low pass filtered image')
        plt.axis('off')
        plt.show()

        # Show the filtered image multiplied with the input image
        plt.imshow(filtered_im_w_input, cmap='gray')
        plt.title('Low pass filtered image')
        plt.axis('off')
        plt.show()

        plt.close('all')

    return filtered_im, filtered_im_w_input


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')
