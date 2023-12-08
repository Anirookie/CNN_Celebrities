# CNN_Celebrities
#Model Architecture:
This architecture includes the application of input layer to a fully-connected neural neetwork
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 126, 126, 32)      896
 max_pooling2d (MaxPooling2D)  (None, 63, 63, 32)       0
 conv2d_1 (Conv2D)           (None, 61, 61, 48)        13872
 max_pooling2d_1 (MaxPooling  (None, 30, 30, 48)       0
 2D)
 conv2d_2 (Conv2D)           (None, 28, 28, 48)        20784
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 48)       0
 2D)
 dropout (Dropout)           (None, 14, 14, 48)        0
 flatten (Flatten)           (None, 9408)              0
 dense (Dense)               (None, 512)               4817408
 dense_1 (Dense)             (None, 128)               65664
 dense_2 (Dense)             (None, 10)                1290
=================================================================
Total params: 4,919,914
Trainable params: 4,919,914
Non-trainable params: 0


#Training and Validation:

For training the dataset we first split the dataset into train and test set with test_size of 80% for training and 20% for testing
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
The batch size is 32 and the training time is 20 epochs.

#Evaluation Metrics:

Model Evalutaion Phase.
2/2 [==============================] - 0s 31ms/step - loss: 1.3689 - accuracy: 0.7647
Accuracy: 76.47

#Documentation:
The model ran successfully resulting in actual and predicted values of the celebrities
          Actual Label    Predicted Label
0              4                4
1              0                0
2              3                1
3              0                0
4              4                4
5              4                4
6              4                4
7              1                1
8              3                1
9              1                1
10             0                0
11             4                2
12             4                3
13             1                1
14             1                1
15             2                2
16             0                0
17             3                3
18             2                2
19             3                3
20             0                0
21             0                0
22             0                2
23             0                1
24             0                0
25             0                2
26             2                2
27             3                1
28             2                2
29             1                1
30             1                1
31             1                1
32             4                4
33             3                3
