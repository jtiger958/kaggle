# Dog vs cat
## Competitoin Description
Discriminate dog and cat by image is traditional task that many researcher suffer. 
The goal of the competitions is distinguish the image of dog and cat and evaluate using binary cross entropy loss function.
We use several backbone network and evaluate each network.

## Score

|              Model              | Epoch | Validation Loss | Validation Accuracy |  Score  |  
|:-------------------------------:|:-----:|:---------------:|:-------------------:|:-------:|  
|      ResNet18 with BCELoss      |   42  |     6.8070-2    |        98.55%       | 0.14888 |  
|      ResNet50 with BCELoss      |   38  |     3.788e-2    |        98.97%       | 0.15173 |  
|      ResNet101 with BCELoss     |   38  |     5.466e-2    |        98.85%       | 0.15305 |  
| ShuffleNet V2 x1.0 with BCELoss |   24  |     7.8121-3    |        98.79%       | 0.08767 |  
|    MobileNet V2 with BCELoss    |   32  |     4.7260-2    |        98.87%       | 0.14321 |  

Competition url: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/