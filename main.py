import numpy as np

class Layer:
    def __init__(self , input_shape , output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(input_shape[1] , output_shape[1])
        self.bias = np.random.randn(1,output_shape[1])

    def forward(self , input):
        self.input = input
        self.output = np.dot(self.input , self.weights) + self.bias
        return self.output

    def backward(self , output_err,learning_rate ):
        layer_error = np.dot(output_err , self.weights.T)
        dweights = np.dot(self.input.T , output_err)

        self.weights = self.weights - dweights * learning_rate
        self.bias = self.bias - output_err * learning_rate

        return layer_error

class ActiveLayer:
    def __init__(self ,input_shape , output_shape , activation , activation_prime):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_err , learning_rate):
        return self.activation_prime(self.input) * output_err

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss_function(self , loss , loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self , input):
        result = []
        n = len(input)
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)

            result.append(output)
        return result


    def train(self , X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0
            outputs = []
            for i in range(len(X_train)):

                #lan truyền tiến
                output = X_train[i]
                for layer in self.layers:
                    output = layer.forward(output)
                lable_array = np.zeros_like(output)
                lable_array[0,np.argmax(output)] = 1
                outputs.append(lable_array)

                #tính hàm mất mát của model
                loss = self.loss(y_train[i] , output)
                total_loss += loss

                #lan truyền ngược
                error = self.loss_prime(y_train[i] , output)
                for layer in reversed(self.layers):
                    error = layer.backward(error , learning_rate)

            average_loss = total_loss / len(X_train)
            total_average_loss = np.mean(average_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_average_loss} , Validation Accuracy: {np.mean(outputs == y_train)}"  )



class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        epsilon = 1e-16
        x = np.clip(x, epsilon, 1 - epsilon)

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        epsilon = 1e-16
        x = np.clip(x, epsilon, 1 - epsilon)

        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0,x)

    @staticmethod
    def relu_prime(x):
        x[x>0] = 1
        x[x<0] = 0
        return x

    @staticmethod
    def mean_squared_error(y_true , y_predict):
        return 0.5 *(y_predict - y_true)**2

    @staticmethod
    def mean_squared_error_prime(y_true , y_predict):
        return y_predict - y_true

    @staticmethod
    def cross_entropy_loss(y_true, y_predict):
        epsilon = 1e-16
        y_predict = np.clip(y_predict, epsilon, 1 - epsilon)

        loss = - (y_true * np.log(y_predict) + (1 - y_true) * np.log(1 - y_predict))
        return loss

    @staticmethod
    def cross_entropy_loss_prime(y_true, y_predict):
        epsilon = 1e-16
        y_predict = np.clip(y_predict, epsilon, 1 - epsilon)

        d_loss = - (y_true / y_predict - (1 - y_true) / (1 - y_predict))
        return d_loss


if __name__ == "__main__":
    # Dữ liệu đào tạo giả định
    X_train = np.array([[[0, 0 , 0]], [[0, 1 , 0]], [[1, 0 , 0]], [[1, 1 , 0]]])
    y_train = np.array([[[1 , 0]], [[1,0]], [[0,1]], [[0,1]]])

    nn = NeuralNetwork()
    nn.add_layer(Layer((1,3) , (1,4)))
    nn.add_layer(ActiveLayer((1,4),(1,4) , ActivationFunctions.sigmoid , ActivationFunctions.sigmoid_prime))
    nn.add_layer(Layer((1,4) , (1,3)))
    nn.add_layer(ActiveLayer((1,3),(1,3) , ActivationFunctions.sigmoid , ActivationFunctions.sigmoid_prime))
    nn.add_layer(Layer((1, 3), (1, 2)))
    nn.add_layer(ActiveLayer((1, 2), (1, 2), ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime))

    nn.set_loss_function(ActivationFunctions.cross_entropy_loss , ActivationFunctions.cross_entropy_loss_prime)

    nn.train(X_train , y_train , 0.01 , 1000)

    out = nn.predict([[0,1,0]])
    print(out)
















