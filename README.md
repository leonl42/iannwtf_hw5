# iannwtf_hw5

### Some parameters of our model:
- datasets (num of elements): train_ds (48.000), valid_ds (12.000), train_ds (10.000) 
- batch_size = 32  
- learning_rate = 0.001
- epochs = 15 
- optimizer: Adam
- some dropout and kernel_regualization (not all layers)

## Receiptive field of our model
In our model we have the following layers:

|         |   type    | kernel/pool size | stride size | padding |
|---------|-----------|------------------|-------------|---------|
| Layer_1 |  Conv2D   |       (5,5)      |    (1,1)    | "same"  |
| Layer_2 |  Conv2D   |       (3,3)      |    (1,1)    | "same"  |
| Layer_3 | MaxPool2D |       (2,2)      |    (2,2)    | "same"  |
| Layer_4 |  Conv2D   |       (9,9)      |    (1,1)    | "same"   |

### Calculating the receiptive field
Our approach will be to calculate the receiptive field size recursively.
So we will first calculate the receiptive field size for Layer_4, then
for Layer_3 and so on.

The formula for calculating the receiptive field size for a higher (earlier)
layer is: s * r + (k - s) where s is the stride of the higher layer, r the receiptive field size 
of the current layer and k the kernel size of the higher layer. Note that we have to do this calculation for
each dimension.

|         | receiptive field size | 
|---------|-----------------------|
| Output  |        (1,1)          |
| Layer_4 |        (8,8)          |
| Layer_3 |        (16,16)        |
| Layer_2 |        (20,20)        |
| Layer_1 |        (24,24)        |

### Positioning of the receiptive field

In order to know the positioning of the receiptive field, we have too
calculate the output sizes of all layers. Due to the padding being "same" everywhere, 
we can just calculate INPUT_SIZE/STRIDE_SIZE for each layer for each dimension.
Our input image has a size of 28x28. 

|         | Input size| Output size |
|---------|-----------|-------------|
| Layer_1 |  (28,28)  |   (28,28)   |
| Layer_2 |  (28,28)  |   (28,28)   |
| Layer_3 |  (28,28)  |   (14,14)   |
| Layer_4 |  (14,14)  |   (14,14)   |

Our output "image" has 14x14 = 196 different output cells.
Each of these cells has a different receiptive field in Layer_1. 
(24,24) is the maximum receiptive field size an output cell can have. 

For example, take the cell at (0,0) from the output image, due to the
padding being "same", this cell will have a rather small receiptive field
due to the padding cells in each layer being part of the field.

![at (0,0)](./img/r_field(0,0).PNG)

The receiptive field of the cell at (6,6) will be a lot bigger on the other
hand, because the receiptive field contains no padding cells on any layer.

![at (6,6)](./img/r_field(6,6).PNG)
