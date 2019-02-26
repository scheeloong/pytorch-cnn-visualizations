"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images

class VanillaBackprop():
    """
    Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode (don't train parameters, batchNorm, dropout behaves differently)
        self.model.eval() 
        # Hook the first layer to get the gradient incoming to the first layer during backward pass. 
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # Set the gradients during the backward pass
            # As the gradients going out of current layer into previous input
            # of size (3, 224, 224)
            self.gradients = grad_in[0] # 
            print("self.gradients.size()", self.gradients.size())
            '''
            print("len(grad_in):", len(grad_in)) # Grad_in is a size 3 tuple
            print("grad_in[0].size()", grad_in[0].size())
            print("grad_in[1].size()", grad_in[1].size())
            print("grad_in[2].size()", grad_in[2].size())
            print("len(grad_out):", len(grad_out)) #
            print("grad_out[0].size():", grad_out[0].size())
            # '''
        # PyTorch allows you to hook functions that are executed per layer
        # either during the forward pass or the backward pass.

        # Get the first layer
        # First layer is known to be closest to input image
        first_layer = list(self.model.features._modules.items())[0][1]
        print("First_layer: ", first_layer)

        '''
        print("self.model", self.model)
        print("self.model.features", self.model.features)
        print("self.model.features._modules", self.model.features._modules)
        print("self.model.features._modules.items()", self.model.features._modules.items())
        print("list(self.model.features._modules.items())[0]", list(self.model.features._modules.items())[0])
        print("list(self.model.features_modulesl.items())[0][1]", list(self.model.features._modules.items())[0][1])
        import sys
        sys.exit(0)
        # '''

        # TODO: How come incoming gradient is nicely the size of image? 
        # NO, it's size of first layer, which assumes input is 224, 224 ??  
        # NO, it's size of input image. TODO: analyze this to find out why!

        # Register hook to the first layer during backward pass
        # So that we can call our own function pointer (callbacks) 
        # during the backward pass to the first layer
        # We want to obtain the incoming gradients to the first layer
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass the model from the input
        model_output = self.model(input_image)

        # Zero gradients everywhere
        # as the model doesn't zero the gradients by default
        # because (optimizer.step() needs the gradients, RNN backprop on shared parameters are cumulative)
        self.model.zero_grad()

        # Target for backprop
        # Set the output to 1 at the target class, and 0 everywhere else. 
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_() # Generate correct shape but all zeroes
        one_hot_output[0][target_class] = 1 # Set the target class to 1

        # Perform a backward pass assuming gradient was 1 from the target class
        # Backward pass
        # Cause, can only call backward after forward 
        # (keeps track of how to perform backward based on path taken during forward pass)
        # The backward pass automatically calls our callback function
        # Our callback function stores the incoming gradients to the first layer
        model_output.backward(gradient=one_hot_output) # Get the gradient with respect to the target class

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224) to (3, 224, 224)
        gradients_as_arr = self.gradients.data.numpy()[0] # Get the gradients and return it
        # Get the gradients
        return gradients_as_arr

if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake

    # Get an example image, pretrained model, pre-process image, it's class, and filename to save into
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model) # Initialized the vanilla backprop.method, hooked to first layer

    # Generate gradients for the current input image and a given target class
    vanilla_grads = VBP.generate_gradients(prep_img, target_class) # Generate the gradients
    
    # Save the gradients obtained as an image

    # Save colored gradients as image
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')

    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)

    # Save grayscale gradients as image
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')

    print('Vanilla backprop completed')
