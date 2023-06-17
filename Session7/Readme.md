# Assignment 7

# First Step - Inital Step

### Model Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

### Objective:

* Get the everything setup correctly
* Set Transforms, DataLoader
* Set Basic Working code
* Set simple training and test logic

### Results:
* Total Parameters: > 5 M

* Best Training Accuracy: 99.67

* Best Test Accuracy: 99.17

### Analysis:
Heavy model which is over-fitting, which need to addressed in another step.


# Basic Model Skeleton

### Model Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
            Conv2d-3           [-1, 64, 24, 24]          18,432
              ReLU-4           [-1, 64, 24, 24]               0
            Conv2d-5          [-1, 128, 22, 22]          73,728
              ReLU-6          [-1, 128, 22, 22]               0
         MaxPool2d-7          [-1, 128, 11, 11]               0
            Conv2d-8           [-1, 32, 11, 11]           4,096
              ReLU-9           [-1, 32, 11, 11]               0
           Conv2d-10             [-1, 64, 9, 9]          18,432
             ReLU-11             [-1, 64, 9, 9]               0
           Conv2d-12            [-1, 128, 7, 7]          73,728
             ReLU-13            [-1, 128, 7, 7]               0
           Conv2d-14             [-1, 10, 7, 7]           1,280
             ReLU-15             [-1, 10, 7, 7]               0
           Conv2d-16             [-1, 10, 1, 1]           4,900
================================================================
Total params: 194,884
Trainable params: 194,884
Non-trainable params: 0
----------------------------------------------------------------

```

### Objective:

* Set Simple Skeleton of Model

### Results:

* Total Parameters: > 194K

* Best Training Accuracy: 98.81

* Best Test Accuracy: 98.47

### Analysis:
Seems overfitting as model is very complex and heavy. Need improvement like lighter model and good accuracy which is next step to work on.


# Lighter Model

### Model Summary:
```
--------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
            Conv2d-3           [-1, 16, 24, 24]           1,440
              ReLU-4           [-1, 16, 24, 24]               0
            Conv2d-5           [-1, 16, 22, 22]           2,304
              ReLU-6           [-1, 16, 22, 22]               0
         MaxPool2d-7           [-1, 16, 11, 11]               0
            Conv2d-8           [-1, 16, 11, 11]             256
              ReLU-9           [-1, 16, 11, 11]               0
           Conv2d-10             [-1, 24, 9, 9]           3,456
             ReLU-11             [-1, 24, 9, 9]               0
           Conv2d-12             [-1, 24, 7, 7]           5,184
             ReLU-13             [-1, 24, 7, 7]               0
           Conv2d-14             [-1, 16, 7, 7]             384
             ReLU-15             [-1, 16, 7, 7]               0
           Conv2d-16             [-1, 10, 1, 1]           7,840
================================================================
Total params: 20,954
Trainable params: 20,954
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.47
Params size (MB): 0.08
Estimated Total Size (MB): 0.55
----------------------------------------------------------------
```

### Objective:

* Make the model lighter

### Results:
* Total Parameters: 20.9k

* Best Training Accuracy: 98.78

* Best Test Accuracy: 98.72

### Analysis:
* Good model which is comparitively lighter model. No overfitting.


# With BatchNorm

### Model Summary:
```

```

### Objective:


### Results:

### Analysis

# With Dropout

### Model Summary:
```

```

### Objective:


### Results:

### Analysis



# GAP

### Model Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             900
              ReLU-6           [-1, 10, 24, 24]               0
       BatchNorm2d-7           [-1, 10, 24, 24]              20
           Dropout-8           [-1, 10, 24, 24]               0
            Conv2d-9           [-1, 10, 22, 22]             900
             ReLU-10           [-1, 10, 22, 22]               0
      BatchNorm2d-11           [-1, 10, 22, 22]              20
          Dropout-12           [-1, 10, 22, 22]               0
        MaxPool2d-13           [-1, 10, 11, 11]               0
           Conv2d-14           [-1, 16, 11, 11]             160
             ReLU-15           [-1, 16, 11, 11]               0
      BatchNorm2d-16           [-1, 16, 11, 11]              32
          Dropout-17           [-1, 16, 11, 11]               0
           Conv2d-18             [-1, 16, 9, 9]           2,304
             ReLU-19             [-1, 16, 9, 9]               0
      BatchNorm2d-20             [-1, 16, 9, 9]              32
          Dropout-21             [-1, 16, 9, 9]               0
           Conv2d-22             [-1, 32, 7, 7]           4,608
             ReLU-23             [-1, 32, 7, 7]               0
      BatchNorm2d-24             [-1, 32, 7, 7]              64
          Dropout-25             [-1, 32, 7, 7]               0
           Conv2d-26             [-1, 10, 7, 7]             320
             ReLU-27             [-1, 10, 7, 7]               0
      BatchNorm2d-28             [-1, 10, 7, 7]              20
          Dropout-29             [-1, 10, 7, 7]               0
        AvgPool2d-30             [-1, 10, 1, 1]               0
================================================================
Total params: 9,490
Trainable params: 9,490
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.04
Estimated Total Size (MB): 0.74
----------------------------------------------------------------
```

### Objective:

* Add the Global Average Pooling

### Results:
* Total Parameters: 9.5k

* Best Training Accuracy: 98.81 (19th Epoch)

* Best Test Accuracy: 99.99 (20th Epoch)

### Analysis:
* GAP not decrease accuracy.
* We are comparing a 14.4k model with 9k model. Since we have reduced model capacity, a reduction in performance is expected.




# With Learning rate Scheduler

### Model Summary:
```

```

### Objective:


### Results:

### Analysis



