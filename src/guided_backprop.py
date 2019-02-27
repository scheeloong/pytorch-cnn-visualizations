"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU

from misc_functions import (get_example_params,
                            convert_to_grayscale, # Convert rgb numpy to grayscale numpy
                            save_gradient_images, # Save numpy images after normalizing to 0 mean, stddev=1
                            get_positive_negative_saliency) # Get only positive gradients, get only negative gradients


class GuidedBackprop():
    """
   Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = [] # Stores the current forward pass's relu's outputs
        # Put model in evaluation mode
        self.model.eval()

        self.update_relus()
        self.hook_layers() # Hook to get gradient from first layer

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # Get the gradients to previous layer from current layers
            self.gradients = grad_in[0]
            print("self.gradients.size():", self.gradients.size())
        # Register hook to the first layer right before the input
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Creates 2 hooks for Relu Activation
        For all layers that are ReLU activation, assign 2 hooks
            1 hook for forward pass of RELU
                - Stores the  output
            1 hook for backward pass
                - 

        Updates relu activation functions so that
        1- stores output in forward pass
        2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last ReLU's forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]

            # Set all output which were positive to 1
            corresponding_forward_output[corresponding_forward_output > 0] = 1

            # 2 things being done here:
            # - DeConvNet => Clamp All negative going back gradients from current relu to 0
            # - Backpropagation => The ReLU's activation which were indeed 0 sets the corresponding gradients to 0
            # => Guided Backpropagation (Striving for simplicity paper)
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output, so later would be update

            # Return the modified gradient output as the gradient output for current layer
            # for remaining backward pass
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU): # only if current layer is a relu layer
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass from current input image to get activations for every class
        model_output = self.model(input_image)

        # Zero gradients for backprop 
        self.model.zero_grad()

        # Target class for backprop with respect to current image
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Backward pass to generate gradients for a given cclass
        model_output.backward(gradient=one_hot_output)

        # Collect the gradients at the input
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
