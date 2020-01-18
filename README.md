# Convolutional Neural Network from scratch [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](http://cnndigits.pythonanywhere.com/)

### [Live Demo](http://cnndigits.pythonanywhere.com/) 

Objective of this work was to write the `Convolutional Neural Network` without using any Deep Learning Library to gain insights of what is actually happening and thus the algorithm is not optimised enough and hence is slow on large dataset like CIFAR-10.
This piece of code could be used for `learning purpose` and could be implemented with trained parameter available in the respective folders for any testing applications like `Object Detection` and `Digit recognition`.<br/>
`It's Accuracy on MNIST test set is above 97%.`


![alt text](images/main.png)


## Architecture

INPUT - CONV1 - RELU - CONV2 - RELU- MAXPOOL - FC1 - OUT

![alt text](images/archi_mnist.jpg)

![alt text](images/archi_cifar.jpg)


## Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* [Numpy](http://www.numpy.org/) - Multidimensioanl Mathematical Computing
* [Matplotlib](https://matplotlib.org/contents.html) - Used to plot Graph
* [Pickle](https://docs.python.org/3/library/pickle.html) - Used to save trained models/object
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Dataset for Digit Recognition
* [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html) - Dataset for Object Recognition

Followings are also required if working/testing on the app.py

* [Flask](http://flask.pocoo.org/) - A microframework for Python
* [Gunicorn](http://gunicorn.org/) - A Python WSGI HTTP Server for UNIX
* [OpenCV](https://docs.opencv.org/trunk/d2/de6/tutorial_py_setup_in_ubuntu.html) -  An open-source library that includes several hundreds of computer vision algorithms

### Directories

- `CIFAR-10 `: Object detection with [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
- `MNIST `: Handwritten Digits Recognition with [MNIST](http://yann.lecun.com/exdb/mnist/)



### Installing

* Clone the repository

```
git clone https://github.com/zishansami102/CNN-from-Scratch
```

* Downlad the dataset and copy that to it's corresponding folder(CIFAR-10/MNIST).
* Move into the required directory (/CNN-from-Scratch/MNIST or /CNN-from-Scratch/CIFAR-10) and then run the following command to start training model

```
python train.py
```

Output:

![alt text](images/training.png)


* To load pre-trained models, change the pickle filename from 'output.pickle' to 'trained.pickle' in run.py: `line No. - 30-31` and comment out the training part form the code in run.py: `line No. - 42-111`

### Application Testing/Running

* In the the directory /CNN-from-Scratch run the following command.

```
python app.py
```
App will start running on the local server [http://127.0.0.1:5000/](http://127.0.0.1:5000/) as shown below : 

![alt text](images/app_running.png)

## Contributing

Mail me at zishansami102@gmail.com if you want to contribute in this project <br/>
My homepage - [zishansami102.github.io](https://zishansami102.github.io/)

## Acknowledgments

* [CS231n.stanford.edu](http://cs231n.stanford.edu/) - Most of the theorotical concepts are taken from here
* [dorajam](https://github.com/dorajam/Convolutional-Network) - Used to gain more concepts
* [Mathematical Concepts](http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/) - Still need to be implemented, but helpful to gain insight
