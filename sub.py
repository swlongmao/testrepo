import numpy,os
import scipy.special

class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, outputnodes,folder_path):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.onodes = outputnodes
        
        self.wih = numpy.loadtxt(folder_path+'\\wih.txt');#numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.loadtxt(folder_path+'\\who.txt');#numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)

        pass   
   
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def main(all_values):
    
    input_nodes = 784 # size of picture is 28*28=784
    output_nodes = 10 # numbers is 10
    file_path  = os.path.abspath(__file__)
    folder_path = os.getcwd()

    n = neuralNetwork(input_nodes,output_nodes,folder_path)# create instance of neural network

    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    
    return label
'''
file_path  = os.path.abspath(__file__)
folder_path1 = os.getcwd().replace('\\','/')
print(folder_path1+'/wih.txt')
'''
