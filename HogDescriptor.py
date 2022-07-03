import cv2
import numpy as np
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def gamma_correction(img):
    return np.power(img / 255.0, 1)


# Render the image and display the calculated gradient direction and gradient amplitude of the image
def render_gradient(image, cell_gradient):
    cell_size = 16
    bin_size = 9
    angle_unit = 180 // bin_size
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(abs(magnitude))))
                angle += angle_gap
    return image


# Get gradient value cell image, gradient direction cell image
def divide_into_cells(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


# Get the gradient direction histogram image, each pixel has 9 values
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[
                                    i, j].flatten())  # .flatten() is a dimensionality reduction function, which reduces the dimensionality to one dimension, the 64 gradient values ​​in each cell are flattened and converted to integers
            ang_list = ang_cell[i, j].flatten()  # Flatten the 64 gradient directions in each cell
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # Amplitude of histogram
            bins[i][j] = binn

    return bins


# Calculate the image HOG feature vector and display
def hog(img, cell_x, cell_y, cell_w):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    print(gradient_magnitude.shape, gradient_angle.shape)
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.imshow() is based on the size of the image, showing the current gradient direction value calculated for each pixel
    # plt.imshow(gradient_magnitude) #Display the gradient value of the image
    plt.imshow(gradient_angle)  # Display the gradient direction value of the image
    # Only one value of the gradient size and direction of the image can be displayed. If plt.imshow() is to be displayed at the same time, it needs to be partitioned.
    plt.show()
    grad_cell = divide_into_cells(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = divide_into_cells(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    hog_image = render_gradient(np.zeros([img.shape[0], img.shape[1]]), bins)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()


def prepare_hog(img):
    cell_w = 8
    print(img.shape)
    x = img.shape[0] - img.shape[0] % cell_w  # Find the number closest to the original image row value divisible by 8
    y = img.shape[1] - img.shape[
        1] % cell_w  # Find the number closest to the original image column value divisible by 8
    resizeimg = cv2.resize(img, (y, x), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("resizeimg", resizeimg)
    cell_x = int(resizeimg.shape[0] // cell_w)  # cell line number
    cell_y = int(resizeimg.shape[1] // cell_w)  # cell column number
    gammaing = gamma_correction(resizeimg) * 255
    feature = hog(gammaing, cell_x, cell_y, cell_w)
    print(feature.shape)
