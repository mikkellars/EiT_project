import numpy as np
import cv2

# 2020-09-29 Developed by Henrik Skov Midtiby

# Load image as greyscale to process.
img = cv2.imread('vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg', 0)
# img = cv2.imread('vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_17.jpg', 0)

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
peak_threshold = 200000 # Suitable for E_3_17
peak_threshold = 800000 # Suitable for E_3_16
shifted_fft_peaks = np.greater(shifted_fft, peak_threshold) * 255.

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
cv2.imwrite("vision/midtiby_method/out00_input_image.png", img)

# Save the shifted fft spectrum.
minval, maxval, a, b = cv2.minMaxLoc(shifted_fft)
cv2.imwrite("vision/midtiby_method/out05_shifted_fft_image.png", 
        shifted_fft * 255 / maxval)

# Save the fft peak mask.
cv2.imwrite("vision/midtiby_method/out07_shifted_fft_peak_mask.png", 
        mask[:, :, 0] * 1.)

# Save the filtered image image.
minval, maxval, a, b = cv2.minMaxLoc(img_back[:, :, 0])
cv2.imwrite("vision/midtiby_method/out10_low_pass_filtered_image.png", 
        img_back[:, :, 0] * 255. / maxval)

# Save the filtered image multiplied with the input image.
minval, maxval, a, b = cv2.minMaxLoc(img * img_back[:, :, 0])
cv2.imwrite("vision/midtiby_method/out20_low_pass_filtered_image.png", 
        img * img_back[:, :, 0] * 255 / maxval)

minval, maxval, a, b = cv2.minMaxLoc(img_back[:, :, 0])
output = img_back[:, :, 0] * 255. / maxval
output = np.where(output > 30, 255, 0)
cv2.imwrite("vision/midtiby_method/output.png", output)