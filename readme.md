#Multilayer Perceptron

##Setting up the Code:

**Compilation**

For compilation you will need to have OpenCL installed and link the project to the correct headers and libraries. Depending on OpenCL Implementation you may need to move a DLL to the folder containing the project.


Your compiler also has to have C++11 enabled to compile the code.


**Execution**

The compiled program uses the following parameters:
You have to use one of these:
- __-p or -parallel__ : OpenCL implementation will be used
- __s or -sequential__: C++ sequential implementation will be used.

It uses a subfolder called "data" for test- and training- image and classification files.

The used files have to be named and stored in the correct folders.
The following paths are used:

- ./data/train-labels.idx1-ubyte
- ./data/train-images.idx3-ubyte

- ./data/t10k-labels.idx1-ubyte

- ./data/t10k-images.idx3-ubyte


The files also have to be IDX files.


**Training Data**

Data that could be used for training and a description of the IDX standard can be found on the MNIST Database webpage.

##Contained Files and usage:

- oclp: Contains the OpenCL implementation of a single hidden layer multilayer perceptron.

- olp: Contains a sequential c++ implementation of a single hidden layer multilayer perceptron.

- nlp: Contains a sequential c++ implementation of a multiple hidden layer multilayer perceptron. This implementation wasn't used often for OCR, the single hidden layer implementation is sufficient.

- idxfile: Contains simple code loading an IDX file into memory. It only supports the unsigned byte datatype.


##License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
