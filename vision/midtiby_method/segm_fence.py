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

    for im, label in zip(images, labels):
        im = load_image(im)
        plt.imshow(im), plt.title('Input image'), plt.axis('off'), plt.show(), plt.close('all')

        # im = contrast_enhancement(im)
        # plt.imshow(im, cmap='gray'), plt.title('Contrast enhacement'), plt.axis('off'), plt.show(), plt.close('all')

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        filtered_im, filtered_im_w_input = frequency_filtering(im, show=True)
        plt.imshow(filtered_im, cmap='gray'), plt.title('Filtered image'), plt.axis('off'), plt.show()

        segm_im = segmentation(filtered_im)
        plt.imshow(segm_im, cmap='gray'), plt.title('Segmented image'), plt.axis('off'), plt.show()

        fence = noise_removal(segm_im)
        plt.imshow(fence, cmap='gray'), plt.title('Fence image'), plt.axis('off'), plt.show()

        # Show ground truth
        label = load_image(label)
        plt.imshow(label, cmap='gray'), plt.title('Grund truth label'), plt.axis('off'), plt.show()


def load_image(path:str, mode='L')->np.array:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    # im = cv2.resize(im, (400, 400), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def frequency_filtering(img:np.array, n_peaks:int=10, show:bool=False)->np.array:
    """By converting the image to frequency domain and using a bandpass filter,
    the frequency of repeated structure in the fence can be filtered out.

    Args:
        im (np.array): Input image.
        n_peaks (int, optional): Number of peaks. Defaults to 10.
        show (bool, optional): If true, then show steps. Defaults to True.

    Returns:
        np.array, np.array: Filtered image, and filtered image with input.
    """

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
    # I get good performance when approx 10 peaks are detected.
    # 200000 - very fine reconstruction of the fence
    # peak_threshold = 200000 # Suitable for E_3_17
    # peak_threshold = 800000 # Suitable for E_3_16
    # shifted_fft_peaks = np.greater(shifted_fft, peak_threshold) * 255.
    for peak_threshold in range(10000, 3000000, 50000):
        shifted_fft_peaks = np.greater(shifted_fft, peak_threshold) * 255.0
        peaks = np.count_nonzero(shifted_fft_peaks)
        if peaks == 16: break
    
    print(f'Peaks: {peaks}')

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

    # Save input image.
    # cv2.imwrite("00_input_image.png", img)

    # Save the shifted fft spectrum.
    minval, maxval, a, b = cv2.minMaxLoc(shifted_fft)
    # cv2.imwrite("05_shifted_fft_image.png", shifted_fft * 255 / maxval)
    shifted_fft_image = shifted_fft * 255 / maxval

    # Save the fft peak mask.
    # cv2.imwrite("07_shifted_fft_peak_mask.png", mask[:, :, 0] * 1.)
    shifted_fft_peak_mask = mask[:, :, 0] * 1.0

    # Save the filtered image image.
    minval, maxval, a, b = cv2.minMaxLoc(img_back[:, :, 0])
    # cv2.imwrite("10_low_pass_filtered_image.png", img_back[:, :, 0] * 255. / maxval)
    low_pass_filtered_image = img_back[:, :, 0] * 255.0 / maxval

    # Save the filtered image multiplied with the input image.
    minval, maxval, a, b = cv2.minMaxLoc(img * img_back[:, :, 0])
    # cv2.imwrite("20_low_pass_filtered_image.png", img * img_back[:, :, 0] * 255 / maxval)
    low_pass_filtered_image = img * img_back[:, :, 0] * 255 / maxval

    if show:
        # Show input image
        plt.imshow(img, cmap='gray')
        plt.title('Input image')
        plt.axis('off')
        plt.show()

        # Show the shifted fft spectrum
        plt.imshow(shifted_fft_image, cmap='gray')
        plt.title('Shifted fft image')
        plt.axis('off')
        plt.show()

        # Show the fft peak mask
        plt.imshow(shifted_fft_peak_mask, cmap='gray')
        plt.title('Shifted fft peak mask')
        plt.axis('off')
        plt.show()

        # Show the filtered image
        plt.imshow(low_pass_filtered_image, cmap='gray')
        plt.title('Low pass filtered image')
        plt.axis('off')
        plt.show()

        # Show the filtered image multiplied with the input image
        plt.imshow(low_pass_filtered_image, cmap='gray')
        plt.title('Low pass filtered image')
        plt.axis('off')
        plt.show()

        plt.close('all')

    return low_pass_filtered_image, low_pass_filtered_image


def contrast_enhancement(im:np.array)->np.array:
    """To improve the contrast in the image, Mahalanobis distance to some predefined
    color of the fence is computed. This operation also prepares for the next step
    providing an output image in gray scale.

    Args:
        im (np.array): Input image.
    """
    low = (50.0, 50.0, 50.0)
    high = (150.0, 150.0, 150.0)
    ret = cv2.inRange(im, low, high)

    # ksize = 3
    # element = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                    (2*ksize + 1, 2*ksize+1),
    #                                    (ksize, ksize))
    # ret = cv2.erode(ret, element)
    # ret = cv2.dilate(ret, element)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ret = cv2.morphologyEx(ret, cv2.MORPH_ERODE, kernel)
    ret = cv2.morphologyEx(ret, cv2.MORPH_DILATE, kernel)

    return ret


def segmentation(im:np.array)->np.array:
    """The image is segmented by Otsu threshold to separate the fence from background
    and to extract a black and white image.

    Args:
        im (np.array): Input image.
    """
    im = im.astype(np.uint8)
    thresh, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return im


def noise_removal(im:np.array)->np.array:
    """Morphological operations remove the noise. Structures are skeletonized and
    dots samller than 10 pixels are removed. Remaining structures are dilated to
    connect and than skeletonized. Structures smaller than 1000 pixels are removed
    and leaving only the fence.

    Args:
        im (np.array): Input image.
    """
    ret = skeletonize(im)

    return ret


def skeletonize(im:np.array)->np.array:
    """Returns a skeletonized verson of the im.
    Reference: http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    Args:
        im (np.array): Input image.

    Returns:
        np.array: Skeletonized image.
    """
    im = im.copy()
    skel = im.copy()
    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.morphologyEx(im, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(im, temp)
        skel = cv2.bitwise_or(skel, temp)
        im[:, :] = eroded[:, :]
        if cv2.countNonZero(im) == 0: break
    return skel


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')
