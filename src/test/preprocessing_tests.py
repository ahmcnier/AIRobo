import unittest
from ..main.image_preprocessing import ImageProcessor


class MyTestCase(unittest.TestCase):
    def test_all_images_being_processed(self):
        imageprocessor = ImageProcessor()
        image_paths = imageprocessor.find_image_file_paths()

        self.assertEqual(len(image_paths), 24989, str(len(image_paths)) + ' images found. Expected 24989 images')
        print('All images expected have been found.')

    def test_check_image_array_size(self):
        imageprocessor = ImageProcessor()
        image_paths = imageprocessor.find_image_file_paths()

        for path in image_paths:
            imageprocessor.obtain_image_dims(path)

        imageprocessor.calculate_mean_image_dims()
        image_arrays = imageprocessor.resize_images(image_paths)
        print(image_arrays.shape)

        self.assertEqual(image_arrays.shape, (24989, 361, 404, 3), 'Image arrays dont match. Shape found ' + str(image_arrays.shape) + ' but expected shape of (24898, 361, 404, 3).')
        print('Correct shape for image array found!')
