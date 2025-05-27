import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

class ImageProcessor:
    def __init__(self, min_img_width=128):
        self.min_img_width = min_img_width
        self.img_width = []
        self.img_height = []
        self.avg_img_width = 0
        self.avg_img_height = 0

    @staticmethod
    def find_image_file_paths(directory_path):
        image_paths = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))

        print("Found " + str(len(image_paths)) + " images in directory " + directory_path + ".")
        return image_paths

    def obtain_image_dims(self, img_file_path):
        width, height = load_img(img_file_path).size

        self.img_width.append(width)
        self.img_height.append(height)

    def calculate_mean_image_dims(self):
        self.avg_img_width = int(np.round(np.mean(self.img_width)))
        self.avg_img_height = int(np.round(np.mean(self.img_height)))

    #this method should resize all of the images to have the mean image heights and widths. Training the CNN requires
    #the image numerical array rather that ++n the raw image
    def resize_images(self, file_paths):
        #set up pytorch tensors
        all_img_arrays = np.zeros((len(self.img_width), self.avg_img_height, self.avg_img_width, 3), dtype='uint8') #dimensions = number of images x height x width x channels

        #loop through all images in file path, resize images and add them to list of image arrays.
        index = 0
        for path in file_paths:
            img = load_img(path, target_size=(self.avg_img_height, self.avg_img_width))
            img_array = img_to_array(img) #convert to array
            img_array = img_array / 255.0 #normalise
            all_img_arrays[index] = img_array
            index += 1

        return all_img_arrays