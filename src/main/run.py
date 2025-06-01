from image_preprocessing import ImageProcessor
import os
from tensorflow.keras.callbacks import EarlyStopping
from classifier import ImageClassifier
import matplotlib.pyplot as plt

def run():
    image_processor = ImageProcessor()
    dataset_bas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    image_paths = image_processor.find_image_file_paths(os.path.abspath(dataset_bas_dir))

    for path in image_paths:
        image_processor.obtain_image_dims(path)

    image_processor.calculate_mean_image_dims()
    print('Creating image arrays...')
    print(dataset_bas_dir)
    train_dataset, val_dataset = image_processor.resize_images('../data')
    print('Image arrays created')

    # #0 = cat
    # #1 = dog
    # #2 = negative

    classifier = ImageClassifier(image_processor.avg_img_height, image_processor.avg_img_height, 3)
    model = classifier.model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=25, callbacks=[early_stop])

    plot_loss_and_accuracy(model_history)



def plot_loss_and_accuracy(model_history):
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['accuracy'], label='Train Accuracy')
    plt.plot(model_history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='Train Loss')
    plt.plot(model_history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('results/train-and-val-fewer-filters.png')


if __name__ == '__main__':
    run()