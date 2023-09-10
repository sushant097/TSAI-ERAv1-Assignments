# Classification of CIFAR-10 dataset

**Objective:**
* To achieve at least 70% accuracy with applying normalization like Group Normalization, Layer Normalization and Batch Normalization.
* Experiment the effect of different normalization techniques applied to the convolutional layers of Neural Network.

**Some Constraints**
1. Less than 50k parameters.
2. At most 20 epochs. 


**My Network Summary**
* Batch Size: 512
* Total Parameters: 37,098
* Image Augmentation applied
    * ColorJitter
    * RandomHorizontalFlip(p=0.3)
    * RandomRotation((-10., 10.))

* Dropout as regularization with probability or 0.02
* 70% as target accuracy obtained in all cases.


```
---------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 32, 32]             270
              ReLU-2           [-1, 10, 32, 32]               0
         GroupNorm-3           [-1, 10, 32, 32]              20
           Dropout-4           [-1, 10, 32, 32]               0
            Conv2d-5           [-1, 10, 32, 32]             900
              ReLU-6           [-1, 10, 32, 32]               0
         GroupNorm-7           [-1, 10, 32, 32]              20
           Dropout-8           [-1, 10, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             160
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 24, 16, 16]           3,456
             ReLU-12           [-1, 24, 16, 16]               0
        GroupNorm-13           [-1, 24, 16, 16]              48
          Dropout-14           [-1, 24, 16, 16]               0
           Conv2d-15           [-1, 16, 16, 16]           3,456
             ReLU-16           [-1, 16, 16, 16]               0
        GroupNorm-17           [-1, 16, 16, 16]              32
          Dropout-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           4,608
             ReLU-20           [-1, 32, 16, 16]               0
        GroupNorm-21           [-1, 32, 16, 16]              64
          Dropout-22           [-1, 32, 16, 16]               0

================================================================
Total params: 37,098
Trainable params: 37,098
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.57
Params size (MB): 0.14
Estimated Total Size (MB): 1.72
----------------------------------------------------------------
```

# Training loss graph


### Training loss graph - Batch Normalization 
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/3e290d4d-cc02-49e7-8664-f868d679d9ad)


### Training loss graph - Group Normalization 
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/c486ae9c-49e4-4c8e-888e-c90b290a608e)


### Training loss graph - Layer Normalization 

![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/fd01bc31-185f-407d-857a-8be05e5ff4b0)


## Mispredicted Image - Batch Normalization 
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/a7e64396-990f-42a7-bc52-8a36b2dfec64)


## Mispredicted Image - Group Normalization 
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/3bc7a50e-cdd7-4f41-a0dd-d73406b01d23)

## Mispredicted Image - Layer Normalization 
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/7477b687-0c06-4676-a70d-728cd8a5a651)
