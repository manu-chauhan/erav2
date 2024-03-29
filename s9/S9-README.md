### The tasks were:

##### Write a new network:

1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
   1. horizontal flip
   2. shiftScaleRotate
   3. coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

#### The approach:

1. Started with basic network but keeping in mind the dimensions of the dataset (CIFAR10) hence minimal kernels from beginning.

2. To keep network efficient and small (less 3x3 later), added 1x1 in betweens to allow mixing of data. Also wanted to keep params as low as possible.

3. Added `dilated` convolution

4. Used `strides` > 1

5. Added `depthwise` convolution in block 4

6. Used Cyclic Learning Rate scheduler for `first 100 epochs`, to allow for frequent movements in loss space and avoid local minima.

7. After a lot of experiments, `base_lr` of 0.003 and `max_lr` of 0.1 was used in Cyclic LR, with `step_up` and `step_down` as 10 and `gamma`=0.85. **Note:** `batch size = 512`

8. `Augmentations` used are:
   1. HorizontalFlip
   2. ShiftScaleRotate
   3. Cutout (max hole=1 and size = 16x16)
   4. HueSaturationValue

9. All the above allowed to touch `70% test accuracy` on `39th epoch` and `train accuracy` of `65.66%`.

10. After `first 100 epochs` (SGD optimizer):
    1. Train accuracy: 77.29 %
    2. Test accuracy: 78.02 % (average test loss = 0.6390)

11. For next 150 epochs used Step_LR, moved to Adam and base lr of 0.005 to shake things, up and restart (resumed) learning, final 87% train and 84.03% test accuracies.

12. For next 150 epochs moved back to SGD (🤦) with Step LR, `step_size=2`, `gamma=0.93`, lr=0.001, momentum=0.9.

13. **Model architecture:**

    ----------------------------------------------------------------        Layer (type)               Output Shape         Param # ================================================================            Conv2d-1           [-1, 32, 32, 32]             864              ReLU-2           [-1, 32, 32, 32]               0       BatchNorm2d-3           [-1, 32, 32, 32]              64            Conv2d-4           [-1, 32, 32, 32]           9,216              ReLU-5           [-1, 32, 32, 32]               0       BatchNorm2d-6           [-1, 32, 32, 32]              64            Conv2d-7           [-1, 32, 17, 17]           9,216              ReLU-8           [-1, 32, 17, 17]               0       BatchNorm2d-9           [-1, 32, 17, 17]              64           Conv2d-10           [-1, 64, 19, 19]           2,304             ReLU-11           [-1, 64, 19, 19]               0      BatchNorm2d-12           [-1, 64, 19, 19]             128           Conv2d-13           [-1, 32, 19, 19]           2,304             ReLU-14           [-1, 32, 19, 19]               0      BatchNorm2d-15           [-1, 32, 19, 19]              64           Conv2d-16           [-1, 32, 19, 19]           1,024           Conv2d-17           [-1, 32, 11, 11]           1,152             ReLU-18           [-1, 32, 11, 11]               0      BatchNorm2d-19           [-1, 32, 11, 11]              64           Conv2d-20           [-1, 64, 13, 13]           1,152             ReLU-21           [-1, 64, 13, 13]               0      BatchNorm2d-22           [-1, 64, 13, 13]             128           Conv2d-23           [-1, 64, 15, 15]             576           Conv2d-24           [-1, 64, 15, 15]           4,096             ReLU-25           [-1, 64, 15, 15]               0      BatchNorm2d-26           [-1, 64, 15, 15]             128           Conv2d-27           [-1, 32, 15, 15]           2,048           Conv2d-28             [-1, 64, 8, 8]           1,152             ReLU-29             [-1, 64, 8, 8]               0      BatchNorm2d-30             [-1, 64, 8, 8]             128           Conv2d-31            [-1, 128, 8, 8]           4,608      BatchNorm2d-32            [-1, 128, 8, 8]             256             ReLU-33            [-1, 128, 8, 8]               0           Conv2d-34            [-1, 128, 8, 8]           2,304           Conv2d-35             [-1, 64, 8, 8]           8,192      BatchNorm2d-36             [-1, 64, 8, 8]             128             ReLU-37             [-1, 64, 8, 8]               0           Conv2d-38             [-1, 64, 4, 4]          36,864           Conv2d-39             [-1, 16, 4, 4]           1,024      BatchNorm2d-40             [-1, 16, 4, 4]              32             ReLU-41             [-1, 16, 4, 4]               0 AdaptiveAvgPool2d-42             [-1, 16, 1, 1]               0           Conv2d-43             [-1, 10, 1, 1]             160 ================================================================ 

    Total params: 89,504 

    Trainable params: 89,504 

    Non-trainable params: 0 ---------------------------------------------------------------- 

    Input size (MB): 0.01 Forward/backward pass size (MB): 3.87 Params size (MB): 0.34 Estimated Total Size (MB): 4.23