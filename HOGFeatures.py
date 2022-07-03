import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2 as cv
import numpy as np
from SpecialImage import SpecialImage


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class HogFeature:
    def get_hog_features(self, data):
        blur = cv.GaussianBlur(data.image, (5, 5), 0)
        fd, hog_image = hog(blur, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=-1, transform_sqrt=True)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        # ax1.axis('off')
        # ax1.imshow(data.image, cmap=plt.cm.gray)
        # ax1.set_title('Input Image')

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # ax2.axis('off')
        # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        # ax2.set_title('Histogram of Oriented Gradients')
        # plt.show()
        special_image = SpecialImage(hog_image_rescaled, data.target)
        special_image.set_fd(fd)
        return special_image
