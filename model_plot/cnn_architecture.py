import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the architecture visualization with improved spacing and text positioning
arch = [
    to_head('../'),
    to_cor(),
    to_begin(),
    
    # Input layer - shape (210, 3)
    to_Conv("input", s_filer=210, n_filer=3, offset="(0,0,0)", to="(0,0,0)", 
            width=1, height=21, depth=1.5, caption="Input\\ (210x3)"),
    
    # First Conv1D layer (8 filters, kernel size 4)
    to_Conv("conv1", s_filer=210, n_filer=8, offset="(3,0,0)", to="(input-east)", 
            width=2, height=21, depth=3, caption="Conv1D\\ (8x4)"),
    to_connection("input", "conv1"),
    
    # MaxPooling1D layer - reduces length to 70
    to_Pool("pool1", offset="(2,0,0)", to="(conv1-east)", width=1, height=7, depth=3, 
            caption="MaxPool\\ 1D(3)"),
    to_connection("conv1", "pool1"),
    
    # Dropout layer
    to_ConvRes("dropout1", s_filer=70, n_filer=8, offset="(2,0,0)", to="(pool1-east)", 
               width=1, height=7, depth=3, opacity=0.2, caption="Dropout\\ (0.1)"),
    to_connection("pool1", "dropout1"),
    
    # Second Conv1D layer (16 filters, kernel size 4)
    to_Conv("conv2", s_filer=70, n_filer=16, offset="(3,0,0)", to="(dropout1-east)", 
            width=2, height=7, depth=5, caption="Conv1D\\ (16x4)"),
    to_connection("dropout1", "conv2"),
    
    # Second MaxPooling1D layer - reduces length to 23
    to_Pool("pool2", offset="(2,0,0)", to="(conv2-east)", width=1, height=2.3, depth=5, 
            caption="MaxPool\\ 1D(3)"),
    to_connection("conv2", "pool2"),
    
    # Second Dropout layer
    to_ConvRes("dropout2", s_filer=23, n_filer=16, offset="(2,0,0)", to="(pool2-east)", 
               width=1, height=2.3, depth=5, opacity=0.2, caption="Dropout\\ (0.1)"),
    to_connection("pool2", "dropout2"),
    
    # Flatten layer - 23 * 16 = 368 neurons
    to_ConvSoftMax("flatten", s_filer=368, offset="(3,0,0)", to="(dropout2-east)", 
                  width=1, height=5, depth=8, caption="Flatten()"),
    to_connection("dropout2", "flatten"),
    
    # First Dense layer (16 neurons)
    to_ConvSoftMax("fc1", s_filer=16, offset="(2.5,0,0)", to="(flatten-east)", 
                  width=2, height=4, depth=4, caption="Dense\\ (16)"),
    to_connection("flatten", "fc1"),
    
    # Third Dropout layer
    to_ConvRes("dropout3", s_filer=16, n_filer=1, offset="(2,0,0)", to="(fc1-east)", 
               width=1, height=4, depth=4, opacity=0.2, caption="Dropout\\ (0.1)"),
    to_connection("fc1", "dropout3"),
    
    # Output layer (2 neurons with softmax)
    to_SoftMax("softmax", s_filer=2, offset="(2,0,0)", to="(dropout3-east)", 
               width=1.5, height=2, depth=2, caption="Dense\\ (2)"),
    to_connection("dropout3", "softmax"),
    
    to_end()
]

def main():
    namefile = "cnn_architecture"
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
