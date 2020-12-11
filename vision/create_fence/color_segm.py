import argparse
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

def load_image(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
    dims = image.shape
    image = image.reshape((-1, 3))
    return image, dims

def quantize_image(image, k):
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    return clt.cluster_centers_.astype(np.uint8)[labels]

def show_quantization(image, shape, k):
    image = image.reshape(shape)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imshow('{} clusters'.format(k), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg')
    args = parser.parse_args()

    image, orig_shape = load_image(args.image)

    for k in reversed(range(10, 15)):
        quant = quantize_image(image, k)
        show_quantization(quant, orig_shape, k)