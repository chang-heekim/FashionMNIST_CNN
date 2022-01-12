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
 
  <b>Implementation</b>
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

</pre>

## AlexNet for FashionMNIST
 | Layer                 | Specification                                                     | 
 | :---------------------| :-----------------------------------------------------------------|
 | Input                 | Channel: 1, Image size: 28 x 28                                   |
 | Convlayer_1           | Channel: 96, kernel_size: 5 x 5, stride: 1, padding: 2            |
 | Activation_2          | ReLU                                                              |
 | MaxPooling_3          | Kernel_size: 3 x 3, stride: 2                                     |
 | LocalResponseNorm_4   | size: 5, alpha: 0.0001, beta: 0.75, k: 2                          |
 | Convlayer_5           | Channel: 256, kernel_size: 5 x 5, stride: 1, padding: 2           |
 | Activation_6          | ReLU                                                              |
 | MaxPooling_7          | Kernel_size: 3 x 3, stride: 2                                     |
 | LocalResponseNorm_8   | size: 5, alpha: 0.0001, beta: 0.75, k: 2                          |
 | Convlayer_9           | Channel: 384, kernel_size: 3 x 3, stride: 1, padding: 1           |
 | Activation_10         | ReLU                                                              |
 | Convlayer_11          | Channel: 384, kernel_size: 3 x 3, stride: 1, padding: 1           |
 | Activation_12         | ReLU                                                              |
 | Convlayer_13          | Channel: 256, kernel_size: 3 x 3, stride: 1, padding: 1           |
 | Activation_14         | ReLU                                                              |
 | MaxPooling_15         | Kernel_size: 3 x 3, stride: 2                                     |
 | Fully_connected_16    | number of neuron: 4096                                            |
 | Activation_17         | ReLU                                                              |
 | Dropout_18            | dropout_prob: 0.5                                                 |
 | Fully_connected_2     | number of nutron: 10                                              |
 | Softmax               | 10 classes                                                        |
  
  <b>Implementation</b>
 <pre>
 class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 96, 5, 1, 2),       # B x 96 x 28 x 28  
            nn.ReLU(),
            nn.MaxPool2d(3, 2),             # B x 96 x 13 x 13
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.Conv2d(96, 256, 5, 1, 2),    # B x 256 x 13 x 13
            nn.ReLU(),
            nn.MaxPool2d(3, 2),             # B x 256 x 6 x 6
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.Conv2d(256, 384, 3, 1, 1),   # B x 384 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),   # B x 384 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),   # B x 384 x 6 x 6
            nn.ReLU(),
            nn.MaxPool2d(3, 2)              # B x 256 x 2 x 2
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.view(x.size(0), -1)

        out = self.classifier(x)
        return out
</pre>
