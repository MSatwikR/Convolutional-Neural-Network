import numpy as np
import sys

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

# Uses One-hot Encoding

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor       
        prediction_tensor = prediction_tensor[np.nonzero(label_tensor)]
        prediction_tensor = prediction_tensor + sys.float_info.epsilon 
        output = -np.sum(np.log(prediction_tensor))
        return output

    def backward(self, label_tensor):
        output = -(label_tensor / self.prediction_tensor)
        return output