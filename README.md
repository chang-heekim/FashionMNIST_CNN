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
 
 <pre>
 class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input = B x 1 x 32 x 32
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),      # B x 6 x 24 x 24   
            nn.Tanh(),
            nn.AvgPool2d(2),            # B x 6 x 12 x 12   
            
            nn.Conv2d(6, 16, 5, 1),     # B x 16 x 8 x 8   
            nn.Tanh(),
            nn.AvgPool2d(2),            # B x 16 x 4 x 4     

            nn.Conv2d(16, 120, 4, 1),   # B x 120 x 1 x 1 
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.view(x.size(0), -1)

        out = self.classifier(x)
        return out
