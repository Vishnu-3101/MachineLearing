import numpy as np
import random
import matplotlib.pyplot as plt

class CrossEntropy:
    @staticmethod
    def cost_function(a,y):
        return np.sum(np.nan_to_num(-(y*np.log(a) + (1-y)*np.log(1-a))))
    
    @staticmethod
    def cost_derivative(z,a,y):
        return (a-y)

class QuadraticCost:
    @staticmethod
    def cost_function(a,y):
        return np.sum((1/2)*(y-a)**2)
    
    @staticmethod
    def cost_derivative(z,a,y):
        return (a-y)*sigmoid_derivative(z)

class Network:
    def __init__(self,net_size,cost = CrossEntropy):
        self.size = net_size
        self.length = len(net_size)
        self.standardWeights()
        self.cost = cost
        self.c = 0
        self.final_cost = []
    
    def standardWeights(self):
        self.weights = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(self.size[1:],self.size[:-1])]
        self.bias = [np.random.randn(x,1) for x in self.size[1:]]
    
    def largeWeights(self):
        self.weights = [np.random.randn(x,y) for x,y in zip(self.size[1:],self.size[:-1])]
        self.bias = [np.random.randn(x,1) for x in self.size[1:]]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.bias, self.weights):
            # print("b: ",np.array(b).shape)
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self,training_data,epochs,mini_size,eta,lamda,test_data = None):
        training_data = list(training_data)
        test_data = list(test_data)
        n_test = len(test_data)
        n = len(training_data)
        n=50000
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_size] 
              for k in range(0,n,mini_size)]
            for mini_data in mini_batches:
                self.update_weights(mini_data,eta,lamda,n)
            self.final_cost.append((i,self.c/n))
            self.c = 0
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), n_test))

    def update_weights(self,mini_data,eta,lamda,n):
        weights = [np.zeros(w.shape) for w in self.weights]
        bias = [np.zeros(b.shape) for b in self.bias]
        for x,y in mini_data:
            n_bias,n_weights = self.backprop(x,y)

        self.weights = [(1-eta*(lamda/n))*w-(eta/len(mini_data))*nw
                        for w, nw in zip(self.weights, n_weights)]

        # self.weights = [w*(1-(eta*lamda/len(mini_data))) - ((eta/len(mini_data))*n_w) for w, n_w in zip(self.weights,n_weights)]
        # for i in n_bias:
        #     print(np.array(i).shape)
        self.bias = [b-(eta/len(mini_data))*nb for b, nb in zip(self.bias, n_bias)]
        # print("After")
        # for i in self.bias:
        #     print(np.array(i).shape)   

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ac = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w,ac) + b
            zs.append(z)
            ac = sigmoid(z)
            activations.append(ac)
        # backward pass

        self.c = self.c + (self.cost).cost_function(activations[-1],y)

        delta = (self.cost).cost_derivative(z[-1],activations[-1], y)
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.length):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def plotGraph(self):
        c_d = []
        epochs = []
        for x,y in self.final_cost:
            epochs.append(x)
            c_d.append(y)
        plt.plot(np.array(epochs),np.array(c_d))
        plt.show()



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))