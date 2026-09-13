# Perceptron Linear Algorithm

## 1. Essentials
The Perceptron Linear Algorithm was introduced by Rosenblatt. It is a supervised learning algorithm used for binary classification. The variables are:
- the inputs $x$: in the case of `MNIST`, these will be grayscale $28 \times 28$ images, 
- labels of the inputs $y$: for `MNIST` these are digits $0-9$, 
- and the weights $w$.

The weights are only updated when the prediction algorithm fails via the rule $w \leftarrow w + y (w \cdot x)$ based on Rosenblatt's proposal. The idea is: When the perceptron fails, the prediction-label product is negative, i.e. $y (x \cdot w) < 0$ , so updating the weights at "pulls" the hyperplane in the direction necessary for the product to become positive. For any reasonable (not linearly separable) dataset, this will not converge.

## 2. Digit selection and filtering
I have opted to make flexible code. 
For filtering: I did not create individual datasets for each digit. This might be optimal in the case of extremely large datasets, but for this one I believed it unnecessary. 
For digit selection: The code is somewhat modular, one can select any target digit and subset of digits to train against by modifying the boilerplate entry point in `training.py`. 

## 3. Code
Around here somewhere.

The images were flattened for easier processing. This also meant that the bias term was integrated into the data and weights vector directly.

## 4. Accuracy
Accuracy was defined to be: "The probability the model will predict the target digit correctly." I'll make some tables here...

## 5. Classification of all digits
The model which is most confident will have the highest prediction $x \cdot w$. One could train multiple "one vs. all" models (one for each digit), and predict the digit based on the highest confidence score. 