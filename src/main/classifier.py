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
            #first layer will get high level features from images (i.e. edges)
            layers.Conv2D(32, (5, 5)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            #this layer will get a little more detail from the images.
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (5, 5)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3), #to reduce overfitting
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2), #to reduce overfitting
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model