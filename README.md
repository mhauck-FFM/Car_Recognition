# Car Recognition
The purpose of these programs is to define and train a neural network using freely available [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data to create a car recognition program. As the model is quite complex with multiple layers, the training process takes a while so that AWS EC2 machines with GPUs are helpful during the training. The final model may then be used to recognize cars on arbitrary pictures passed to the application.
<br> The development of this program was part of [this](https://www.udemy.com/course/deep-learning-und-ai/) online course.

### Program Structure

- `CIFAR10_Car_Recognition_Model.py`, contains the model definition and training
- `CIFAR10_Car_Recognition_Application.py`, contains the main program that uses the pretrained model to recognize cars on given pictures
- `carrecognition.h5`, a pre-trained version of the model for application

#### Setup Instructions

- If using the pre-trained model, parse the file path to the model as well as your desired picture to `CIFAR10_Car_Recognition_Application.py` and run it.
- If training of the model is desired, specify the path where the model should be stored after training in `CIFAR10_Car_Recognition_Model.py` and run it. After completion, specify the path to the model and your desired picture in `CIFAR10_Car_Recognition_Application.py` and run it. The program should return your picture with cars highlighted.

#### Required Pathon Modules

- numpy
- keras
- pathlib
- PIL
- matplotlib
