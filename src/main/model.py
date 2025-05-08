import fiftyone as fo
import PIL as Image
import os

print(fo.zoo.load_zoo_dataset("coco-2017").classes)

negative_dataset = fo.zoo.download_zoo_dataset(
    "coco-2017",
    classes=['bicycle', 'bus', 'tie', 'skateboard', 'fork', 'apple', 'laptop']
    )

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'negative-image-samples'))

counter = 0

for picture in negative_dataset:
    image = Image.open(picture)
    image.save(os.path.join(base_dir, counter + '.png'))
    bg_text_file = open('../bg.txt', 'a')
    bg_text_file.write(base_dir + '/' + counter + '.png\n')
    bg_text_file.close()



negative_dataset.name = "random-negative-samples"
negative_dataset.persistent = True

session = fo.launch_app(negative_dataset)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

test_path = os.path.join(base_dir, 'test')
train_path = os.path.join(base_dir, 'train')

test_cat_path = os.path.join(test_path, 'cats')
test_dog_path = os.path.join(test_path, 'dogs')
train_cat_path = os.path.join(train_path, 'cats')
train_dog_path = os.path.join(train_path, 'dogs')

positive_test_data = [f for f in os.listdir(test_cat_path)]
positive_test_data.extend([f for f in os.listdir(test_dog_path)])

positive_train_data = [f for f in os.listdir(train_cat_path)]
positive_train_data.extend([f for f in os.listdir(train_dog_path)])