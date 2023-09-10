# Classification of CIFAR-10 dataset

**Objective:**
* Learn about advance convolution and augmentation

**Some constraints:**
1. Achieve 85% accuracy
2. Total RF must be more than 44

3. One of the layers must use Depthwise Separable Convolution and one of the layers must use Dilated Convolution

4. Use GAP (compulsory)

5. Use albumentation library and apply:

   5.1. horizontal flip

   5.2. shiftScaleRotate

   5.3. coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

6. Achieve 85% accuracy

7. Total Params to be less than 200k


**My Network Summary**
* Batch Size: 512
* Total Parameters: 134,722
* Image Augmentation applied
    * horizontal flip
    * Shift Scale Rotate
    * Coarse Dropout

* Dropout as regularization with probability or 0.02
* 85% as target accuracy obtained in all cases.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 32, 32]             270
              ReLU-2           [-1, 10, 32, 32]               0
       BatchNorm2d-3           [-1, 10, 32, 32]              20
           Dropout-4           [-1, 10, 32, 32]               0
            Conv2d-5           [-1, 24, 32, 32]           2,160
              ReLU-6           [-1, 24, 32, 32]               0
       BatchNorm2d-7           [-1, 24, 32, 32]              48
           Dropout-8           [-1, 24, 32, 32]               0
            Conv2d-9           [-1, 32, 30, 30]           6,912
           Conv2d-10           [-1, 64, 30, 30]          18,432
             ReLU-11           [-1, 64, 30, 30]               0
      BatchNorm2d-12           [-1, 64, 30, 30]             128
          Dropout-13           [-1, 64, 30, 30]               0
           Conv2d-14           [-1, 32, 32, 32]           2,048
             ReLU-15           [-1, 32, 32, 32]               0
      BatchNorm2d-16           [-1, 32, 32, 32]              64
          Dropout-17           [-1, 32, 32, 32]               0
           Conv2d-18           [-1, 32, 17, 17]           4,096
           Conv2d-19           [-1, 64, 17, 17]          18,432
             ReLU-20           [-1, 64, 17, 17]               0
      BatchNorm2d-21           [-1, 64, 17, 17]             128
          Dropout-22           [-1, 64, 17, 17]               0
           Conv2d-23           [-1, 96, 17, 17]          55,296
             ReLU-24           [-1, 96, 17, 17]               0
      BatchNorm2d-25           [-1, 96, 17, 17]             192
          Dropout-26           [-1, 96, 17, 17]               0
           Conv2d-27             [-1, 64, 9, 9]          24,576
           Conv2d-28             [-1, 64, 9, 9]             576
             ReLU-29             [-1, 64, 9, 9]               0
      BatchNorm2d-30             [-1, 64, 9, 9]             128
          Dropout-31             [-1, 64, 9, 9]               0
           Conv2d-32           [-1, 16, 11, 11]           1,024
             ReLU-33           [-1, 16, 11, 11]               0
      BatchNorm2d-34           [-1, 16, 11, 11]              32
          Dropout-35           [-1, 16, 11, 11]               0
        AvgPool2d-36             [-1, 16, 1, 1]               0
           Conv2d-37             [-1, 10, 1, 1]             160
================================================================
Total params: 134,722
Trainable params: 134,722
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 5.78
Params size (MB): 0.51
Estimated Total Size (MB): 6.30
----------------------------------------------------------------
```

**Transformations**
```
train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=.45, mask_fill_value = None),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)
test_transforms = A.Compose(
    [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ]
)
```

## 85% Test Accuracy achieved consistently after 43th epochs
![image](https://github.com/mapillary/inplace_abn/assets/30827903/34f60f9e-7b0d-43d6-a0cc-e764b49dddf2)

# Training loss graph

![image](https://github.com/mapillary/inplace_abn/assets/30827903/0dd8d704-7f76-4fb7-ace2-ee49b89ce692)



## Mispredicted Image 
![image](https://github.com/mapillary/inplace_abn/assets/30827903/b9dc3353-6cf1-4011-9fa9-3066781e72de)



