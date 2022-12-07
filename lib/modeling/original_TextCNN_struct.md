The Text CNN struct:
```
ActionIntentionDetection(
  (bbox_embedding): Sequential(
    (0): Linear(in_features=4, out_features=256, bias=True)
    (1): ReLU()
  )
  (conv3): Conv2d(1, 1, kernel_size=(3, 256), stride=(1, 1))
  (conv4): Conv2d(1, 1, kernel_size=(4, 256), stride=(1, 1))
  (conv5): Conv2d(1, 1, kernel_size=(5, 256), stride=(1, 1))
  (conv6): Conv2d(1, 1, kernel_size=(6, 256), stride=(1, 1))
  (conv7): Conv2d(1, 1, kernel_size=(8, 256), stride=(1, 1))
  (conv8): Conv2d(1, 1, kernel_size=(10, 256), stride=(1, 1))
  (Max3_pool): MaxPool2d(kernel_size=(28, 1), stride=(28, 1), padding=0, dilation=1, ceil_mode=False)
  (Max4_pool): AvgPool2d(kernel_size=(27, 1), stride=(27, 1), padding=0)
  (Max5_pool): AvgPool2d(kernel_size=(26, 1), stride=(26, 1), padding=0)
  (Max6_pool): AvgPool2d(kernel_size=(25, 1), stride=(25, 1), padding=0)
  (Max8_pool): AvgPool2d(kernel_size=(23, 1), stride=(23, 1), padding=0)
  (Max10_pool): AvgPool2d(kernel_size=(21, 1), stride=(21, 1), padding=0)
  (linear1): Linear(in_features=6, out_features=60, bias=True)
)
```
