#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "nlp.h"
#include "olp.h"
#include "idxfile.h"

float calcMeanSquaredError(const int pSize,const float* pTarget, const float* pResult)
{
    float error = 0;
    for(int i=0; i<pSize; ++i)
    {
        error += (pResult[i]-pTarget[i])*(pResult[i]-pTarget[i]);
    }
    return error/pSize;
}

int findHighestIndex(const float* pResults, const int pSize)
{
    int highestIndex = 0;
    float highestValue = 0.0f;
    for(int i=0;i<pSize;++i)
    {
        if(pResults[i]>highestValue)
        {
            highestIndex = i;
            highestValue = pResults[i];
        }
    }
    return highestIndex;
}

void trainOCR()
{
    IDXFile trainLabels("./data/train-labels.idx1-ubyte" );
    IDXFile *trainImages = new IDXFile("./data/train-images.idx3-ubyte" );

    IDXFile testLabels("./data/t10k-labels.idx1-ubyte" );
    IDXFile *testImages = new IDXFile("./data/t10k-images.idx3-ubyte" );

    //test if all files have been correctly read and have the right sizes
    if(!trainLabels.hasError() && trainLabels.getDimensionNumber() == 1 &&
       !trainImages->hasError() && trainImages->getDimensionNumber() == 3 &&
       !testLabels.hasError() && testLabels.getDimensionNumber() == 1 &&
       !testImages->hasError() && testImages->getDimensionNumber() == 3 &&
       trainImages->getDimensions()[1]*trainImages->getDimensions()[2] ==
       testImages->getDimensions()[1]*testImages->getDimensions()[2])
    {
        const unsigned int imageSize        = trainImages->getDimensions()[1]*trainImages->getDimensions()[2];
        std::cout << "Preparing images." << std::endl;
        float **fTrainImages = new float*[trainImages->getDimensions()[0]];
        for(unsigned int i=0;i<trainImages->getDimensions()[0];++i)
        {
            fTrainImages[i] = new float[imageSize];
            for(unsigned int j=0;j<imageSize;++j)
            {
                fTrainImages[i][j] = *(trainImages->getDataPointer()+i*imageSize+j);
            }
        }
        trainImages->deleteData();

        float **fTestImages = new float*[testImages->getDimensions()[0]];
        for(unsigned int i=0;i<testImages->getDimensions()[0];++i)
        {
            fTestImages[i] = new float[imageSize];
            for(unsigned int j=0;j<imageSize;++j)
            {
                fTestImages[i][j] = *(testImages->getDataPointer()+i*imageSize+j);
            }
        }
        testImages->deleteData();

        std::cout << "OCR-Training." << std::endl;
        constexpr unsigned int samples      = 10;
        constexpr float eta                 = 0.025f;
        const unsigned int hiddenNodes[]    = {300};
        const unsigned int hiddenLayers     = 1;
        std::cout << "using images with: " << imageSize << " pixels." << std::endl;
        std::cout << "using mlp with: ";
        for(unsigned int i=0; i<hiddenLayers; ++i)
        {
            std::cout << hiddenNodes[i];
            if(i<hiddenLayers-1)
            {
                std::cout << ",";
            }
        }
        std::cout << " hidden nodes and eta: " << eta << "." << std::endl;

        std::cout << "setting up data." << std::endl;
        float targets[samples][samples]={0};
        for(unsigned int i=0; i<samples; ++i)
        {
            targets[i][i]=1.0f;
        }

        OneLayerPerceptron mlp(eta,imageSize,hiddenNodes[0],samples)/*(eta,imageSize,hiddenLayers,hiddenNodes,samples)*/;

        float error     = 1.0f;
        const unsigned int testTotal = testImages->getDimensions()[0];
        while(error > 0.05)
        {
            std::cout << "training the mlp." << std::endl;
            for(unsigned int iterations=0; iterations<1; ++iterations)
            {
                for(unsigned int i=0; i<trainImages->getDimensions()[0]; ++i)
                {
                    const int targetIndex = (int)trainLabels.getDataPointer()[i];
                    mlp.train(fTrainImages[i],targets[targetIndex]);
                }
            }
            std::cout << "classifying test data." << std::endl;
            int correct = 0;
            for(unsigned int i=0; i<testTotal; ++i)
            {
                const int targetIndex = (int)testLabels.getDataPointer()[i];
                float* tmpResults = mlp.classify(fTestImages[i]);
                if(findHighestIndex(tmpResults,10)==targetIndex)
                {
                    ++correct;
                }
                delete [] tmpResults;
            }
            error = (1-correct/(float)testTotal);
            std::cout << "error: " << error << "." <<std::endl;
        }

        //free used memory
        for(unsigned int i=0;i<trainImages->getDimensions()[0];++i)
        {
            delete [] fTrainImages[i];
        }
        delete [] fTrainImages;
        for(unsigned int i=0;i<testImages->getDimensions()[0];++i)
        {
            delete [] fTestImages[i];
        }
        delete [] fTestImages;
    }
    else
    {
        std::cerr << "required files appear to contain errors" << std::endl;
    }
}

inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static const char source[] =
    "kernel void add(\n"
    "       ulong n,\n"
    "       global const double *a,\n"
    "       global const double *b,\n"
    "       global double *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "       c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";

int OCLTest() {
    const size_t N = 1 << 20;

    try {
        // Get list of OpenCL platforms.
        std::vector<cl::Platform> platform;
        cl::Platform::get(&platform);

        if (platform.empty())
        {
            std::cerr << "OpenCL platforms not found." << std::endl;
            return 1;
        }

        cl::Context context;
        std::vector<cl::Device> device;
        for(auto p = platform.begin(); device.empty() && p != platform.end(); p++)
        {
            std::vector<cl::Device> pldev;

            try
            {
                p->getDevices(CL_DEVICE_TYPE_CPU, &pldev);

                for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++)
                {
                    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

                    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

                    device.push_back(*d);
                    context = cl::Context(device);
                }
            }
            catch(...)
            {
                device.clear();
            }
        }

        if (device.empty())
        {
            std::cerr << "no usable device found." << std::endl;
            return 1;
        }

        std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create command queue.
        cl::CommandQueue queue(context, device[0]);

        // Compile OpenCL program for found device.
        cl::Program program(context, cl::Program::Sources( 1, std::make_pair(source, strlen(source))));

        try {
            program.build(device);
        }
        catch (const cl::Error&)
        {
            std::cerr << "OpenCL compilation error" << std::endl;
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0]) << std::endl;
            return 1;
        }

        cl::Kernel add(program, "add");

        // Prepare input data.
        std::vector<double> a(N, 1);
        std::vector<double> b(N, 2);
        std::vector<double> c(N);

        // Allocate device buffers and transfer input data to device.
        cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(double), a.data());
        cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(double), b.data());
        cl::Buffer C(context, CL_MEM_READ_WRITE, c.size() * sizeof(double));

        // Set kernel parameters.
        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        // Launch kernel on the compute device.
        queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);

        // Get result back to host.
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());

        // Should get '3' here.
        std::cout << c[42] << std::endl;
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return 1;
    }
    return 0;
}

int main()
{
    return OCLTest();
}
