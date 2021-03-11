from random import random
import math

class NeuralNetwork:

    # Initiate weights and biases
    def __init__(self, inputs, layer, outputs):
        self.inputs = inputs
        self.layer = []
        for i in range(layer):
            self.layer.append([[], 0, 0, 0])
            for j in range(inputs):
                self.layer[i][0].append(random() * 2 - 1)
        self.outputs = []
        for i in range(outputs):
            self.outputs.append([[], 0])
            for j in range(layer):
                self.outputs[i][0].append(random() * 2 - 1)
    
    # Activation function
    def activation(self, x):
        return 1 / (1 + math.exp(-x))
    def dActivation(self, x):
        a = self.activation(x)
        return a * (1 - a)
    
    # Forward propagation
    def prop(self, inputs):
        outputs = []
        for i in range(len(self.layer)):
            node = self.layer[i]
            s = 0
            for j in range(len(inputs)):
                s += node[0][j] * inputs[j]
            s += node[1]
            self.layer[i][2] = self.activation(s)
        
        for i in range(len(self.outputs)):
            node = self.outputs[i]
            s = 0
            for j in range(len(self.layer)):
                s += node[0][j] * self.layer[j][2]
            s += node[1]
            outputs.append(self.activation(s))
        
        return outputs
    
    # Backpropagation
    def backprop(self, inputs, desired, strength = 1):
        outputs = []
        for i in range(len(self.outputs)):
            outputs.append([0, 0])
        cost = 0
        
        for i in range(len(self.layer)):
            s = 0
            for j in range(self.inputs):
                s += self.layer[i][0][j] * inputs[j]
            s += self.layer[i][1]
            self.layer[i][2] = self.activation(s)
            self.layer[i][3] = s
        
        for i in range(len(self.outputs)):
            s = 0
            for j in range(len(self.layer)):
                s += self.outputs[i][0][j] * self.layer[j][2]
            s += self.outputs[i][1]
            outputs[i][0] = self.activation(s)
            outputs[i][1] = s
        
        for i in range(len(outputs)):
            o = outputs[i]
            difference = o[0] - desired[i]
            cost += difference * difference
            derivative = 2 * difference * self.dActivation(o[1]) * strength
            for j in range(len(self.layer)):
                l = self.layer[j]
                derivative2 = self.outputs[i][0][j] * self.dActivation(l[3])
                for k in range(self.inputs):
                    self.layer[j][0][k] -= derivative * derivative2 * inputs[k]
                self.layer[j][1] -= derivative * derivative2
                self.outputs[i][0][j] -= derivative * l[2]
            self.outputs[i][1] -= derivative
        
        return cost



