from tensorflow.keras import layers, models, regularizers

class ImageClassifier():
    def __init__(self, img_height, img_width, n_channels):
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        self.num_classes = 3 #cats, dogs and negative samples (empty images)

    def model(self):
        model = models.Sequential([
            layers.Input((self.img_height, self.img_width, self.n_channels)),
            layers.Conv2D(32, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (4, 4), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model