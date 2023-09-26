# Image_Classification\


Image Classification of Heartrate Images using CNN

Introduction:-

The image classification of Heartrate images using CNN Model. To train the model to heartrate images(ECG)  of different classes that are present in the dataset are Left Bundle Branch block, Normal , Premature Ventricular Contraction, Premature Atrial contraction, Right Bundle Branch Block, Ventricular Fibrillation. Build Convolution Neural Network and external bias is added, and Dense layers are added based on the accuracy of the model. For this model we consider TensorFlow, Keras, Numpy, pandas, OpenCV and different libraries are added. For building these models  create requirements.txt  in the requirement.txt add the required libraries are OpenCV ,numpy, pandas, TensorFlow, Keras, split-folder and sk-learn.
For install the requirement.txt open the vscode. In vscode open the new terminal and type this command pip install requirements.txt for using this command to install all the libraries in the requirements folder.

Extensions:-

		Add the required extensions in vscode to run the codes and create the python file(.py) and Jupyter(.ipynb) and many more . click on the extensions left side and add the python ,pylance, jupyter, Azure, git, python indent and based on your requirement.

Project:-

Importing  the required libraries to image classification model.
Numpy works with the arrays, and it also works with the functions are algebra, Fourier Transformer and metrices. Numpy documentation click here.
Pandas are DataFrame. It is used to do data manipulation, data analysis, loading and saving the data in DataFrame format and statistical analysis and many things. Pandas’ documentation click here.
Matplotlib is a visualization library. To visualize the graphs like line graphs and bar graphs and also used to show the images. Matplotlib documentation click here.
TensorFlow and Keras are the main libraries and all the Deep Learning models and different layers and Activation functions and Optimizers are present in those two libraries to see the documentation of Tensorflow and Keras.
Importing the libraries of numpy as “np”, pandas as “pd”, OpenCV as “cv2”, Keras and TensorFlow.

Layers and Models:-

Importing the required layers and models. For importing the layers from  Keras and TensorFlow Libraries and models also from TensorFlow and Keras. For image classification model we consider Sequential, Dense Layer, Convolution2d Layer, MaxPooling Layer, Flatten Layer. The Sequential model “from keras.models import Sequential” and load_model and all the layers (from keras.layers import (required layer)) .  from keras.preprocessing import image  and from keras.utils import (img_to_array,  load_img). 
 It allows us to specify the neural network like Convolution Neural Network, Artificial Neural Network and Recurrent Neural Network . The Sequential is an empty space we can create n number of layers in the sequential. Sequential as the input to output. We build the required number of layers between the input and output.
keras.model has two types of models that are available in keras one is sequential model and another one is model class used with functional API.
keras.layers means in the keras library as different number of layers are present in keras for using this keras.layers importing the required layers.   
From keras.preprocessing.image import imagedatagenerator. The keras imagedatagenerator class provides a quick and easy way to augment your images and different augmentation techniques like standardization, rotation, shifts, flips, rescale, shear_range, zoom_range.
We need to import the ImageDataGenerator. It is used to change the original data to random data and rescale the image. The default image pixel should be 1 to 255 and divided by the pixels of 255 and the image data should be the 0 to 1 format. Apply the rescale to Train Dataset and Test Dataset.

Importing the image from keras.preprocessing techniques. The Keras is the module they are so many preprocessing techniques from this technique importing of image is used to read the image into workspace. 
The load_model is used to load the full convolution Neural Network when we saved after completed the model.


Convolution2d:-  

The first layer of a Convolutional Neural Network is always a Convolution layer. Convolutional layers apply a convolution operation to the input, passing the result to the next layer. A convolution converts all the pixels in its receptive field into a single value. For example, if you would apply a convolution to an image, you will be decreasing the image size as well as bringing all the information in the field together into a single pixel. The final output of the convolutional layer is a vector. Based on the type of problem we need to solve and on the kind of features we are looking to learn, we can use different kinds of convolutions.
A convolutional neural network is a feed-forward neural network that is generally used to analyze visual images by processing data with grid-like topology. It’s also known as ConvNet. A convolutional neural network is used to detect and classify objects in an image.
The convolution neural network is to reduce the images into a form of easier to process, without losing any features that are important for getting a good prediction. For example, consider an image of height and width is 4*4 and consider the kernel of 2*2 then image is multiplied with the kernel the image size should be 3*3 the figure is shown in below.

The kernel multiply with the first four boxes in the convolution layer and the calculation is  0*5 + -1*3 + 6*1 + 2*7 = 17.

The kernel multiply with the second four boxes in the convolution layer and the calculation is  0*3 + -1*1 + 1*7 + 2*8 = 32. Do this calculation to the full convolution layer then the image should reduce . considering the kernel size based on your requirement. The kernel values of 0,-1,1,2 are not the kernel values those are only for example. The real kernel values are taken by system.
	Convolution2d(32,(3,3)) means the 32 is the features we extracting from image of convolution2d and (3,3) means the 3*3 kernel is multiply with the convolution2d.lets take an example.


For example we consider the 6*6 convolution2d than multiply with the 3*3 kernel the multiplication process explain in the above, after that the image should be resized into the 4*4 convolution.
	The Input_shape(64,64,3) means the input image of the 64 *64 pixels and the 3 means the 3 channels of RGB.
	The convolution2d is added into sequential. One layer is added and after that we add the pooling layer.

Activation Function:-

Activation function is the crucial part in the model is used to control how well the network model learns the training dataset. Activation function is used to determine whether the neural network is activated or not . There are so many activation functions to choose which activation function is based on the model. For the CNN model most used activation functions are  sigmoid , Relu ,Tanh , Softmax.

	Activation function gives the output of the neural network in between 0 to 1 or -1 to 1 that is dependent on the function used. There are two types of activation functions, one is linear activation function and non-linear activation function. The linear activation function the output is not confined between the range and the non-linear activation function mostly used the activation functions like sigmoid, Relu, Tanh and many more. 
	We consider the Relu activation function we use the Relu in hidden layer to avoid vanishing the gradient problem and Relu is non-linear activation function. The Relu is the activation function is commonly used.


	Softmax activation function:-  `
	The Softmax activation scales the numbers into probabilities. The Softmax activation function converts the vector of number into vector of probabilities.
 
	The Softmax activation give the probabilities of each class. which class as the more probability that class will be the output.


Pooling Layer:-

	They two types of Pooling Layers and one is the MaxPooling Layer and another one is the AvgPooling Layer.

MaxPooling Layer:-

The MaxPooling layer is used to take the output of convolution layer and consider the MaxPooling layer of 2*2 matrix they are different pooling layers like AvgPooling  and MaxPooling layer. By considering which Pooling layer check the which pooling layer gives the best accuracy. 

When we apply the MaxPooling layer to the output from the convolution layer when we consider the first four boxes then MaxPooling layer takes the value is 9 which is the max value in those four boxes.

We consider the second four boxes. Then MaxPooling layer takes value is 7 which is the max value in those four boxes. Lets continue the process for all boxes present in the output from the convolution layer.

	Pooling_size(2,2) means size of the MaxPooling layer. After you add the pooling layer, the image has resized. This pooling layer added next to the convolution2d.

AvgPooling Layer:- 

	The AvgPooling layer is used to take the output of convolution layer and consider the AvgPooling layer of 2*2 matrix they are different pooling layers like AvgPooling  and MaxPooling layer. By considering which Pooling layer check the which pooling layer gives the best accuracy. 
	When we apply the AvgPooling layer to the output from the convolution layer when we consider the first four boxes then AvgPooling layer takes the value is 5 which is the Avg value in those four boxes.

	We consider the second four boxes. Then AvgPooling layer takes value is 3.5 which is the Avg value in those four boxes. Lets continue the process for all boxes present in the output from the convolution layer.

	Pooling_size(2,2) means size of the AvgPooling layer. After you add the pooling layer, the image has resized. This pooling layer added next to the convolution2d.
Added the another convolution2D to extract the features from the output of MaxPooling layer after that added the another MaxPooling layer resize the image.

For adding all those layers to extract the features and reduce the size of the image. The four layers are added first layer is convolution2d and second layer MaxPooling layer and third layer convolution2d and fourth layer is MaxPooling layer, In this sequence first layer extract the key features after the resize the image in second layer and from the resized image the third layer extract the features after that fourth layer resize the image this is the process of CNN model and how many layers should added is based on the project your doing and based on the accuracy of the model we need to add the number of layers. After all this layers we add the flatten layer.

Kernel:-

	A kernel is a matrix and multiplied with the input image such that the output should be enhanced and the kernel is used to resized the input image and the kernel size should be mention on your own and the kernel size should be 2 * 2 matrix or 3 * 3 matrix or many more we should consider the kernel size based on the model. The matrix value taken by the system.

Flatten Layer:-

	Flatten layer is used to convert the 2-dimensional array  into a continuous linear vector. Commonly used to transition convolution layer to the fully connected layer. Flatten takes the output from the Pooling layer and gives the input to the fully connected layer.


Dense Layer :-

	After completed all the layers the single linear vector is connected to the fully connected neural network is also dense layer. The dense layer means takes the input from the output of precise layers. We can add n number of dense layers. I am adding 6 Dense layers of activation function is Relu. Unit 128 means the output of the layers and the main point the last dense layer units should be the number of class present in your project the activation function should be SoftMax. SoftMax activation function scales the numbers to probabilities of each class.
And the last Dense layer of units should be the 6 because of this project as 6 class are present. Activation function should be the SoftMax activation function.

From the above image is a fully connected layer and blue circles are the output from the flatten layer and gives to the fully connected layer and between the blue and green circles the weights are initialized. Dense layers are added to the output from the flatten layer and In fully connected layer we add the six dense layers in the model. activation function in the dense layer is Relu of the five dense layers and the last dense layer of activation function is Softmax because the Softmax activation function give the probabilities of each class and units is 128 for the first five layers and the units should taken by the requirements of the model and the last layer units should be the 6 and 6 the is number of class present in the model.
	The basic fully connected neural network is shown above from the above green circles are the dense layer they are two layers are connected, the first layer of  units are 6 and the second layer as 4 units. And the three classes are the output. For our project 6 dense layers are those green circles add 6 layers and units is 128 and 128 green circle are present in the first layer same for all the 4 layers and the last layer as units is 6 and 6 green circles are present in those layers. 

model.summary() is means to check the how many layers are added to this model and number of dense layers and check the sequence of the layers are add to the model. To check the output shape height and width and features are shown from the MaxPooling layer the image is resized.

After adding all the layers to the model and compile the model, for compiling the model we take the optimizer, loss function, metrics.

Optimizers:-

	Optimizers are used to adjust the weights in the fully connected neural network. There are so many optimizers present. Weights should be initialized in the fully connected neural network. Let’s take an example.


Based on the weights the check the neuron is activated are not and the weights should be taken by system and some default weights should be initialize by 
the system based on those weights we check the accuracy and if the accuracy is fine its ok then accuracy is not good, we need to change the weights for this changing of weights we added the optimizers, and these optimizers change the weights until the accuracy is good. We consider the Adam Optimizer.

Adam Optimizer:-

	Adam optimizers focus on the hyperparameter they are so many parameters present. Adam mainly focus on the learning rate. The learning rate should be minimum; the point should be reaching the global minimum.

Loss Function:- 

	Loss function means how well the model is performing and compare the actual output and the predicted output. There are two types of loss function in classification problem.
•	Binary cross entropy/log loss.
•	Categorical cross entropy.

Binary cross entropy is used to binary class classification problem that means the output should be the 0 or 1 format in this classification only two class are present.

      Categorical cross entropy is used to multi-class classification problem that means the output should be the n number of class. This project as multi-class classification that’s why we consider the categorical cross entropy.

Accuracy:- 

	Accuracy is used to predict the how good the model is worked. As you can see, Accuracy can be easily described using the Confusion matrix terms such as True Positive, True Negative, False Positive, and False Negative.

After compilation as done and model.fit_generator to run the model give the training data to generator and steps_per_epoch means number of images taken for the 1 iteration for example we have 1000 images for steps_per_epoch is 100 images 10 iteration are taken for this project we consider the all the images and only iteration is taken same as the validation data and epochs means the full cycle of the training data and we consider 9 epochs that means the 9 times we run the full cycle and how many epochs should consider based on the accuracy and you should consider any number of epochs and if the accuracy is goes constant at the point you brake the epochs. In which number accuracy is more we should consider that number as epochs. Just check the last three epochs and the accuracy is 0.9594,0.9647,0.9692.

After load the model let’s take an image to predict and path of an image is taken into the data_path and load the image and the target_size is 64,64 let’s convert the image into the array format and next expand the features or dimensions and after the convert the image to an array then we pass into the model. The image should be given to the model to predict the image belongs to which class. The model gives the probability to each class and which class as the highest probability that is the output that’s why we consider the np.argmax. Argmax is used to which class as the maximum probability and that class should be print as output. And class 2 as the maximum probability and consider the index of your class.










