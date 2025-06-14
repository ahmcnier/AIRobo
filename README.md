# AIRobo
First AI Robotics Project

I aim to make a pet teasing robot toy. The robot should be able to chase the pet and/or run away from the pet. 

I will have combine object detection models and deep learning to detect the pet. An ultrasound sensor will be fitted to detect proximity of the pet. 

I have used a dataset consisting of dog and cat pictures (sourced from https://github.com/laxmimerit/dog-cat-full-dataset.git), in an attempt to reduce overfitting the model towards one specific animal or type of animal. I have sourced extra "negative" image samples from the kaggle (https://www.kaggle.com/datasets/prasunroy/natural-images).

I first used a pretrained YOLO model to verify my dataset successfully. I next, looked to train my own model based on the Haar Cascade architecture. 

I have all the training of the model in the 'src/model' directory. This accesses the train and validation data in the 'src/data' directory. I set up some test cases in the 'src/test' directory. You can find the rapsberry pi code in the 'src/raspberry-pi' directory.