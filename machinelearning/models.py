from torch import stack, tril
from torch.nn import Module
import torch


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim, exp

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()

        self.dimension = dimensions
        weight = zeros(1, dimensions)
        self.w = Parameter(weight)


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return tensordot(x, self.get_weights())

        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self(x)

        if score >= 0:
            return 1
        else: return -1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        self.linear1 = Linear(1, 100)
        self.linear2 = Linear(100, 100)
        self.linear3 = Linear(100, 1)

        self.act = relu
   

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = self.linear1(x)
        x = relu(x)

        x = self.linear2(x)
        x = relu(x)

        x = self.linear3(x)
        
        return x


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        
        self.linear1 = Linear(input_size, 256)
        # self.linear2 = Linear(512, 256)
        self.linear3 = Linear(64, 10)
        self.linear4 = Linear(256, 64)
        # self.linear5 = Linear(128, 64)

        self.act = relu
        self.softmax = softmax


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x = self.linear3(self.act(self.linear4(self.act(self.linear1(x)))))
        return self.softmax(x)



class LanguageIDModel(Module):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        
        # Use a larger hidden size as suggested in instructions
        hidden_size = 256
        
        self.Wx = Linear(self.num_chars, hidden_size)
        self.Wh = Linear(hidden_size, hidden_size)
        
        self.fc1 = Linear(hidden_size, 128)
        self.out = Linear(128, len(self.languages))

    def forward(self, xs):
        h = None
        
        for char in xs:
            if h is not None:
                # f(h_i, x_i) = g(x_i * Wx + h_i * Wh)
                h = torch.relu(self.Wx(char) + self.Wh(h))
            else:
                # f_initial(x_0) = g(x_0 * Wx)
                h = torch.relu(self.Wx(char))
        
        return self.out(self.fc1(h))



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    output_height = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    output_width = input_tensor_dimensions[1] - weight_dimensions[1] + 1
    Output_Tensor = zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            region = input[i:i+weight_dimensions[0], j:j+weight_dimensions[1]]
            Output_Tensor[i, j] = torch.sum(region * weight)
    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.input_size = 28 * 28
        self.output_size = output_size
        self.convolution_weights = Parameter(ones((3, 3)))
        self.fc1 = Linear(676, 128)
        self.act = relu
        self.fc2 = Linear(128, output_size)

    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        K = self.k_layer(input)  # (B,T,D)
        Q = self.q_layer(input)  # (B,T,D)
        V = self.v_layer(input)  # (B,T,D)

        QK = matmul(Q, movedim(K, -1, -2)) / (self.layer_size**0.5)  # (B,T,T)
        QK = QK.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))[0]

        attention = softmax(QK, dim=-1)
        out = matmul(attention, V)  # (B,T,D)
        return out

        """YOUR CODE HERE"""

