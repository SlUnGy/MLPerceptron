#Multilayer Perceptron

This program will train a Multilayer Perceptron for Optical Character Recognition. The Multilayer Perceptron will be trained for recognizing the digits zero to nine.  
By supplying command line parameters you can use either a sequential C++ implementation or a parallel OpenCL implementation for training. The program will try to reach over 95% correct estimates and then terminate.

##Using the Code:

**Compilation**

For compilation you will need to have OpenCL installed and link the project to the correct headers and libraries. Depending on the OpenCL Implementation you use you may need to move an OpenCL.dll to the folder containing the binary.

Your compiler also **has to have C++11** enabled to compile the code.

**Execution**

The compiled program uses the following parameters:
You **have to** use one of these:
- *-p or -parallel* : OpenCL implementation will be used
- *-s or -sequential*: C++ sequential implementation will be used.

It uses a subfolder called "data" for test- and training- image and classification files.

The used files have to be named and stored in the correct folders.
The following paths are used:

- `./data/train-labels.idx1-ubyte`
- `./data/train-images.idx3-ubyte`
- `./data/t10k-labels.idx1-ubyte`
- `./data/t10k-images.idx3-ubyte`


The files also have to conform to the IDX file standard, which is described on the MNIST Database webpage.


**Training Data**

Data that could be used for training can be found on the MNIST Database webpage. The images are assumed to be greyscale images ( each pixel has a value from 0 to 255).

##Contained Files and usage:

- oclp: Contains the host part of the OpenCL implementation of a single hidden layer multilayer perceptron.  
- mlp.cl: Contains the client OpenCL implementation.  
- olp: Contains a sequential c++ implementation of a single hidden layer multilayer perceptron.  
- nlp: Contains a sequential c++ implementation of a multiple hidden layer multilayer perceptron. This implementation wasn't used often for OCR, the single hidden layer implementation is sufficient.  
- idxfile: Contains simple code loading an IDX file into memory. It only supports the unsigned byte datatype.  


##License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
