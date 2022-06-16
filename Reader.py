import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import os

from SpecialImage import SpecialImage


class Reader:
    @staticmethod
    def read_from(path):
        dataset = []
        dataset_path = path

        for file_name in os.listdir(dataset_path):
            if ".tiff" in file_name:
                img = cv2.imread(os.path.join(dataset_path, file_name))
                dataset.append(img)
        dataset = np.array(dataset)

        # training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=25)

        training_data = np.array(dataset)
        testing_data = np.array(dataset)
        print(training_data.shape)
        print(testing_data.shape)
        return training_data, testing_data

    @staticmethod
    def get_target_for_image(file_name):
        if "AN" in file_name:
            return 0
        if "DI" in file_name:
            return 1
        if "FE" in file_name:
            return 2
        if "HA" in file_name:
            return 3
        if "NE" in file_name:
            return 4
        if "SA" in file_name:
            return 5
        if "SU" in file_name:
            return 6

    @staticmethod
    def read_from_with_target(path):
        dataset = []
        dataset_path = path
        for file_name in os.listdir(dataset_path):
            if ".tiff" in file_name:
                img = cv2.imread(os.path.join(dataset_path, file_name))
                target = Reader.get_target_for_image(file_name)
                special_image = SpecialImage(img, target)
                dataset.append(special_image)

        return dataset

