# Funding Analysis using Deep Learning and Neural Networks

#### Overview:
The purpose of this analysis was to develop a machine learning model, specifically a neural network, that can predict the success of organizations applying for funding from Alphabet Soup, a nonprofit foundation. By identifying the most promising applicants, Alphabet Soup aims to allocate its resources more effectively, maximizing the impact of its funding on the organizations it supports.

To achieve this goal, the Alphabet Soup business team has provided a dataset containing information on over 34,000 organizations that have received funding from Alphabet Soup in the past. 

The neural network models tested below were designed using TensorFlow and Keras using multiple hidden layers with varying numbers of neurons and activation functions.  The model is then compiled, trained on the training dataset, and evaluated using the test dataset to determine its loss and accuracy.

#### Results:
##### Data Preprocessing:
* Target variable
  * IS_SUCCESSFUL
* Feature variables
  * APPLICATION_TYPE
  * AFFILIATION
  * CLASSIFICATION
  * USE_CASE
  * ORGANIZATION
  * STATUS
  * INCOME_AMT
  * SPECIAL_CONSIDERATIONS
  * ASK_AMT

In the dataset, the target variable is **IS_SUCCESSFUL**. This is the variable we are trying to predict using the features in the dataset. The other variables (excluding EIN and NAME) are the feature variables that will be used as inputs for the model to learn the relationship between the features and the target variable.

#### Compiling, Training, and Evaluating the Model
The tests below were conducted to achieve an optimized neural network model for the given classification problem.  None of these tests achieved the target performance of 75%.  Based on these results, however, Model 4 with the Leaky ReLU activation function, 100 epochs, and 2 hidden layers (80 and 20 neurons) performed the best in terms of accuracy.  Even so, it is important to note that the differences in accuracy between the models are relatively small.

##### Attempt 1: 
ReLU activation function, 100 epochs, 2 hidden layers with 80 and 20 neurons.
**Loss: 0.5603, Accuracy: 0.7287**
![1-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_1.png)

##### Attempt 2: 
ReLU activation function, 100 epochs, 2 hidden layers with 120 and 30 neurons.
**Loss: 0.5624, Accuracy: 0.7261**
* *Increased the number of neurons in each layer to explore if a larger model improves accuracy. However, it slightly decreased accuracy.*
![2-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_2.png)

##### Attempt 3: 
ReLU activation function, 120 epochs, 2 hidden layers with 120 and 30 neurons.
**Loss: 0.5679, Accuracy: 0.7284**
* *Increased the number of epochs to test if additional training improves the model's performance. The accuracy was slightly better than Model 2, but still lower than Model 1.*
![3-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_3.png)

##### Attempt 4: Leaky ReLU activation function, 100 epochs, 2 hidden layers with 80 and 20 neurons.
**Loss: 0.558, Accuracy: 0.7317**
* *Tested an alternative activation function, Leaky ReLU. This model achieved the highest accuracy among the tested models.*
![4-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_4.png)

##### Attempt 5: 
Leaky ReLU activation function, 120 epochs, 2 hidden layers with 80 and 20 neurons.
**Loss: 0.5609, Accuracy: 0.7299**
* *Increased the number of epochs with the Leaky ReLU activation function to test if additional training improves performance. The accuracy was slightly lower than Model 4.*
![5-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_5.png)

##### Attempt 6 (Keras Tuner optimized): tanh activation function, 100 epochs, 3 hidden layers with 9, 1, and 5 neurons.
**Loss: 0.5578, Accuracy: 0.7261**
* *Tested a model with a different activation function (tanh) and an additional hidden layer, as recommended by the Keras Tuner. This model showed the lowest loss value but had lower accuracy compared to Models 4 and 5.*
![6-Results](https://github.com/rchascall/deep-learning-challenge/blob/main/images/attempt_6.png)

#### Technologies used:
1.	Python
2.	Pandas
3.	Scikit-learn (sklearn)
4.	TensorFlow
5.	Keras
6.	Keras Tuner

#### Summary:
Further optimization and testing of different activation functions, and training configurations might be necessary to achieve higher accuracy. Additionally, other types of models, such as Random Forest or Support Vector Machines, could provide a comparison to the neural network models' performance on this classification problem.
