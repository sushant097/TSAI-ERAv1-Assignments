# CIFAR-10 Dataset Classisfication with custom ResNet and One Cycle Policy Learning rate

## Summary of the Model:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
              ReLU-2           [-1, 64, 32, 32]               0
       BatchNorm2d-3           [-1, 64, 32, 32]             128
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
              ReLU-7          [-1, 128, 16, 16]               0
       BatchNorm2d-8          [-1, 128, 16, 16]             256
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
             ReLU-11          [-1, 128, 16, 16]               0
      BatchNorm2d-12          [-1, 128, 16, 16]             256
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
             ReLU-15          [-1, 128, 16, 16]               0
      BatchNorm2d-16          [-1, 128, 16, 16]             256
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
             ReLU-19          [-1, 256, 16, 16]               0
      BatchNorm2d-20          [-1, 256, 16, 16]             512
          Dropout-21          [-1, 256, 16, 16]               0
        MaxPool2d-22            [-1, 256, 8, 8]               0
             ReLU-23            [-1, 256, 8, 8]               0
      BatchNorm2d-24            [-1, 256, 8, 8]             512
          Dropout-25            [-1, 256, 8, 8]               0
           Conv2d-26            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-27            [-1, 512, 4, 4]               0
             ReLU-28            [-1, 512, 4, 4]               0
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
          Dropout-30            [-1, 512, 4, 4]               0
           Conv2d-31            [-1, 512, 4, 4]       2,359,296
             ReLU-32            [-1, 512, 4, 4]               0
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
          Dropout-34            [-1, 512, 4, 4]               0
           Conv2d-35            [-1, 512, 4, 4]       2,359,296
             ReLU-36            [-1, 512, 4, 4]               0
      BatchNorm2d-37            [-1, 512, 4, 4]           1,024
          Dropout-38            [-1, 512, 4, 4]               0
        MaxPool2d-39            [-1, 512, 1, 1]               0
           Conv2d-40             [-1, 10, 1, 1]           5,120
================================================================
Total params: 6,573,632
Trainable params: 6,573,632
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.50
Params size (MB): 25.08
Estimated Total Size (MB): 34.59
----------------------------------------------------------------
```

## Image Augmentation
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/9f8f448f-b195-476d-9e36-c8c9cee16ee0)

### LR Finder Output:
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/bf19439c-2cb1-4a7e-bd3b-0438c2dc4e4b)

## One Cycle Learning Rate
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/8d84ed77-baee-4621-85db-89e9001a4260)

## Training output Logs
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/f4a6e3c1-8d78-4e2c-ab5d-a02976a555f2)

## Wrong Prediction
![image](https://github.com/sushant097/Deep-Learning-Paper-Scratch-Implementation/assets/30827903/95d664d5-8851-41c3-af78-0a53e339e979)


