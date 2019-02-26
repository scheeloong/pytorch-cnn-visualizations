"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

# Save image saves a numpy image into an image file
# preprocess_image converts a PIL image to PyTorch Variable Tensor for training
# recreate_image converts the trained PyTorch Variable Tensor back to a PIL image
from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
    Produces an image that minimizes the loss of a convolution
    operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer # 17
        self.selected_filter = selected_filter # 5
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')
        print("self.model:", self.model)

    def hook_layer(self):
        # For forward, should be hook_function(module, input, output)
        # Input is the input to current layer, output is the output to current layer
        # Set the output as the outgoing forward pass from 0 all the way to selected filter
        # As don't need output after that filter
        def hook_function(module, inputToCurrentLayer, outputOfCurrentLayer): 
            # Gets the conv output of the selected filter (from selected layer)
            # Get the outgoing output to a given neuron
            # The filter is one of the output depths
            self.conv_output = outputOfCurrentLayer[0, self.selected_filter] 
            print("len(inputToCurrentLayer)", len(inputToCurrentLayer))
            print("len(outputOfCurrentLayer)", len(outputOfCurrentLayer))
            print("inputToCurrentLayer[0].size()", inputToCurrentLayer[0].size())
            print("outOfCurrentLayer[0].size()", outputOfCurrentLayer[0].size())
            print("self.selected_filter", self.selected_filter)
            print("self.conv_output.size()", self.conv_output.size())

        # Hook the selected layer of the convnet for forward pass
        self.model[self.selected_layer].register_forward_hook(hook_function) # 

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()

        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))

        # Process image and return variable
        processed_image = preprocess_image(random_image, False)

        # Define optimizer for the image, where the image is a variable
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)

        #for i in range(1, 31):
        for i in range(1, 2): # TODO: TEMP FOR DEBUGGING  AND LEARNING
            optimizer.zero_grad() # Zero the generated gradients for this iteration

            # Assign create image to a variable to move forward in the model
            x = processed_image

            # Keep forward passing till reach the layer of interest
            # then, will trigger the hooked function call
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Now, have the convolution output at the layer of interest for the output depth of that layer


            # Want to minimize mean of the output of selected layer's filter

            # THe output are real numbers
            # The higher the value (more positive) => More activation
            # If you take the mean or sum, it just accounts for all position
            # Now if you minimize the negative of the mean, it's same as maximizing
            # Therefore, it's about maximizing the output
            # It's about finding the input image that maximizes the output!

            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter

            loss = -torch.mean(self.conv_output)

            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward pass from the current loss tensor
            loss.backward() # Generate the gradients using a single backward pass

            # Update image
            optimizer.step() # Update the parameters using the generated gradients

            # Save image
            if i % 5 == 0:
                # Recreate image into a PIL tensor
                self.created_image = recreate_image(processed_image)
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        #for i in range(1, 31):
        for i in range(1, 2): # TEMP FOR DEBUGGING  AND LEARNING
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features # Get VGG16 parameters
    # Note: you need to print the pretrained_model
    # then select the layer that belongs to a CONV layer
    cnn_layer = 17
    filter_pos = 5
    '''
    print(pretrained_model) # Printed output below
    self.model: Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace)
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): ReLU(inplace)
      (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (27): ReLU(inplace)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    '''
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
