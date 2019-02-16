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
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # Set the gradients during the backward pass
            # As the gradients coming into the current layer
            self.gradients = grad_in[0] 

        # PyTorch allows you to hook functions that are executed per layer
        # either during the forward pass or the backward pass.

        # Get the first layer
        # First layer is known to be closest to input image
        first_layer = list(self.model.features._modules.items())[0][1]

        '''
        print("self.model", self.model)
        print("self.model.features", self.model.features)
        print("self.model.features._modules", self.model.features._modules)
        print("self.model.features._modules.items()", self.model.features._modules.items())
        print("list(self.model.features._modules.items())[0]", list(self.model.features._modules.items())[0])
        print("list(self.model.features_modulesl.items())[0][1]", list(self.model.features._modules.items())[0][1])
        import sys
        sys.exit(0)
        '''

        # Register hook to the first layer during backward pass
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_() # Generate correct shape but all zeroes
        one_hot_output[0][target_class] = 1 # Set the target class to 1
        # Backward pass
        model_output.backward(gradient=one_hot_output) # Get the gradient with respect to the target class
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        # Get the gradients
        return gradients_as_arr

if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)

    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')

    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)

    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')

    print('Vanilla backprop completed')
