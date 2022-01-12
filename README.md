# FashionMNIST_CNN
  - learning/validation was conducted using Google Colab.
  - use LeNet-5, AlexNet for FashionMNist Dataset
 
## FashionMNIST Dataset
  - number of Train Datasets : 54000
  - number of Validation Datasets: 6000
  - number of Test Datasets: 10000
  
  - 10 classes
    - 0: 'T-shirt/top',
    - 1: 'Trouser',
    - 2: 'Pullover',
    - 3: 'Dress',
    - 4: 'Coat',
    - 5: 'Sandal',
    - 6: 'Shirt',
    - 7: 'Sneaker',
    - 8: 'Bag',
    - 9: 'Ankle boot' 
    - <img src='https://github.com/chang-heekim/FashionMNIST_CNN/blob/main/images/image.png'/>

## LeNet-5 for FashionMNIST
 | Layer                 | Specification                                                     | 
 | :---------------------| :-----------------------------------------------------------------|
 | Input                 | Channel: 1, Image size: 28 x 28                                   |
 | Convlayer_1           | Channel: 6, kernel_size: 5 x 5, stride: 1, padding: 0             |
 | Activation            | Tanh                                                              |
 | AveragePooling_1      | Kernel_size: 2 x 2                                                |
 | Convlayer_2           | Channel: 16, kernel_size: 5 x 5, stride: 1, padding: 0            |
 | Activation            | Tanh                                                              |
 | AveragePooling_2      | Kernel_size: 2 x 2                                                |
 | Convlayer_3           | Channel: 120, kernel_size: 4 x 4, stride: 1, padding: 0           |
 | Activation            | Tanh                                                              |
 | Fully_connected_1     | number of neuron: 84                                              |
 | Activation            | Tanh                                                              |
 | Fully_connected_2     | number of nutron: 10                                              |
 | Softmax               | 10 classes                                                        |
