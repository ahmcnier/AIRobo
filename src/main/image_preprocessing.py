import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        # self.avg_img_width = int(np.round(np.mean(self.img_width)))
        # self.avg_img_height = int(np.round(np.mean(self.img_height)))
        self.avg_img_width = 256
        self.avg_img_height = 256

    #this method should resize all of the images to have the mean image heights and widths. Training the CNN requires
    #the image numerical array rather than the raw image and one-hot coded vectors for labels - both done automatically #
    #through the ImageDataGenerator method from tensorflow
    def resize_images(self, dir):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_image_generator = datagen.flow_from_directory(
            dir,
            target_size=(self.avg_img_height, self.avg_img_width),
            batch_size=16,
            class_mode='categorical',
            subset='training'
        )

        val_image_generator = datagen.flow_from_directory(
            dir,
            target_size=(self.avg_img_height, self.avg_img_width),
            batch_size=16,
            class_mode='categorical',
            subset='validation'
        )

        return train_image_generator, val_image_generator