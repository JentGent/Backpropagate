// Neural Network
function NeuralNetwork(inputs, hiddens, outputs) {
    this.inputs = inputs;
    this.layer = [];
    this.outputs = [];
    for(var i = 0; i < hiddens; i += 1) {
        this.layer.push([[], 0]);
        for(var j = 0; j < inputs; j += 1) {
            this.layer[this.layer.length - 1][0].push(Math.random() * 2 - 1);
        }
    }
    for(var i = 0; i < outputs; i += 1) {
        this.outputs.push([[], 0]);
        for(var j = 0; j < this.layer.length; j += 1) {
            this.outputs[this.outputs.length - 1][0].push(Math.random() * 2 - 1);
        }
    }
}
// Activation function (sigmoid)
NeuralNetwork.prototype.activation = function(x) {
    return 1 / (1 + Math.exp(-x));
    // return max(x, 0);
    // return Math.tanh(x);
    // return x > 0 ? x : x * 0.1;
};
NeuralNetwork.prototype.dActivation = function(x) {
    var a = this.activation(x);
    return a * (1 - a);
    // return x >= 0 ? 1 : 0;
    // return 1 - sq(Math.tanh(x));
    // return x >= 0 ? 1 : 0.1;
};
// Forward propagation, returns outputs
NeuralNetwork.prototype.prop = function(inputs) {
    var outputs = [];
    for(var i = 0; i < this.outputs.length; i += 1) {
        outputs.push(0);
    }
    for(var i = 0; i < this.layer.length; i += 1) {
        var sum = 0;
        for(var j = 0; j < inputs.length; j += 1) {
            sum += this.layer[i][0][j] * inputs[j];
        }
        this.layer[i][2] = this.activation(sum + this.layer[i][1]);
    }
    for(var i = 0; i < this.outputs.length; i += 1) {
        var sum = 0;
        for(var j = 0; j < this.layer.length; j += 1) {
            sum += this.outputs[i][0][j] * this.layer[j][2];
        }
        sum = this.activation(sum + this.outputs[i][1]);
        outputs[i] = sum;
    }
    return outputs;
};
// Back propagation, performs gradient descent and returns cost
NeuralNetwork.prototype.backprop = function(inputs, desired, strength) {
    var cost = 0;
    strength = strength || 1;
    var outputs = [];
    for(var i = 0; i < this.outputs.length; i += 1) {
        outputs.push([0, 0]);
    }
    for(var i = 0; i < this.layer.length; i += 1) {
        var sum = 0;
        for(var j = 0; j < inputs.length; j += 1) {
            sum += this.layer[i][0][j] * inputs[j];
        }
        this.layer[i][3] = sum + this.layer[i][1];
        this.layer[i][2] = this.activation(this.layer[i][3]);
    }
    for(var i = 0; i < this.outputs.length; i += 1) {
        var sum = 0;
        for(var j = 0; j < this.layer.length; j += 1) {
            sum += this.outputs[i][0][j] * this.layer[j][2];
        }
        outputs[i][1] = sum + this.outputs[i][1];
        sum = this.activation(outputs[i][1]);
        outputs[i][0] = sum;
    }
    
    for(var i = 0; i < outputs.length; i += 1) {
        var o = outputs[i];
        cost += (o[0] - desired[i]) * (o[0] - desired[i]);
        var derivative = 2 * (o[0] - desired[i]) * this.dActivation(o[1]) * strength;
        for(var j = 0; j < this.layer.length; j += 1) {
            var l = this.layer[j];
            var derivative2 = this.outputs[i][0][j] * this.dActivation(l[3]);
            for(var k = 0; k < this.inputs; k += 1) {
                this.layer[j][0][k] -= derivative * derivative2 * inputs[k];
            }
            this.layer[j][1] -= derivative * derivative2;
            this.outputs[i][0][j] -= derivative * l[2];
        }
        this.outputs[i][1] -= derivative;
    }
    return cost;
};