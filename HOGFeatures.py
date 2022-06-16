import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2 as cv

from HogDescriptor import Hog_descriptor
from SpecialImage import SpecialImage


class HogFeature:
    def get_hog_features(self, data):
        blur = cv.GaussianBlur(data.image, (5, 5), 0)
        fd, hog_image = hog(blur, orientations=8, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=True, channel_axis=-1, transform_sqrt=True)

        # hog1 = Hog_descriptor(blur, cell_size=8, bin_size=8)
        # fd, hog_image = hog1.extract()

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #
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
