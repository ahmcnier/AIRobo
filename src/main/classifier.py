from tensorflow.keras import layers, models

class ImageClassifier():
    def __init__(self, img_height, img_width, n_channels):
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        self.num_classes = 3 #cats, dogs and negative samples (empty images)

    def model(self):
        model = models.Sequential([
            layers.Input((self.img_height, self.img_width, self.n_channels)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model