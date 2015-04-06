#include "oclp.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <random>

OpenCLPerceptron::OpenCLPerceptron(const float pEta, const int pInputPerceptrons, const int pHiddenPerceptrons, const int pOutputPerceptrons)
    :m_foundDevice{false}, m_sourceFile{"mlp.cl"}, m_eta{pEta}, m_hidPerceptrons{pHiddenPerceptrons},
     m_inpPerceptrons{pInputPerceptrons}, m_outPerceptrons{pOutputPerceptrons},
     m_hidOutput(m_hidPerceptrons),m_hidWeights((m_hidPerceptrons)*(m_inpPerceptrons+1)),
     m_outWeights(m_outPerceptrons*(m_hidPerceptrons+1))
{
    randomizeWeights();
}

OpenCLPerceptron::~OpenCLPerceptron()
{

}


void OpenCLPerceptron::randomizeWeights()
{
	/*
        doesn't work on mingw&windows....
        std::random_device rd;
    */
	std::mt19937 mt(time(NULL));
	std::uniform_real_distribution<> distribution(-1, 1);

    for (int i=0;i<m_hidPerceptrons*(m_inpPerceptrons+1);i++)
    {
		m_hidWeights[i] = distribution(mt);
    }

    for (int i=0;i<m_outPerceptrons*(m_hidPerceptrons+1);i++)
    {
		m_outWeights[i] = distribution(mt);
	}
}

bool OpenCLPerceptron::initOpenCL()
{
    try{
        std::vector<cl::Platform> allPlatforms;
        cl::Platform::get(&allPlatforms);

        if (allPlatforms.empty())
        {
            std::cerr << "OpenCL platforms not found." << std::endl;
            return false;
        }

        for(auto currentPlatform = allPlatforms.begin();
            !m_foundDevice && currentPlatform != allPlatforms.end();
            currentPlatform++)
        {
            std::vector<cl::Device> allDevices;
            currentPlatform->getDevices(CL_DEVICE_TYPE_CPU, &allDevices);

            for(auto currentDevice = allDevices.begin();
                !m_foundDevice && currentDevice != allDevices.end();
                currentDevice++)
            {
                if (currentDevice->getInfo<CL_DEVICE_AVAILABLE>())//add other selection criteria here
                {
                    m_device.push_back(*currentDevice);
                    m_context = cl::Context(m_device);
                    m_foundDevice = true;
                }
            }
        }

        if (!m_foundDevice)
        {
            std::cerr << "no usable device found." << std::endl;
            return false;
        }

        std::cout << "using:" << m_device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

        std::ifstream file(m_sourceFile);
        std::string prog( std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));

        m_program = cl::Program(m_context, cl::Program::Sources( 1, std::make_pair(prog.c_str(), prog.length()+1)));

        try {
            m_program.build(m_device);
        }
        catch (const cl::Error&)
        {
            std::cerr << "compilation error: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device[0]) << std::endl;
            return false;
        }

        m_calcHiddenOutput  = cl::Kernel(m_program, "calcHidden");
        m_classify          = cl::Kernel(m_program, "calcOut");
        m_calculateDelta    = cl::Kernel(m_program, "calcDelta");
        m_backpropagation   = cl::Kernel(m_program, "backprop");
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return false;
    }
    return true;
}

bool OpenCLPerceptron::initTraining(std::vector<float> *trainImg, std::vector<float> *trainClf)
{
    if(trainImg != nullptr && trainClf != nullptr)
    {
        try {
            m_trImg     = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, trainImg->size() * sizeof(float), trainImg->data());
            m_hWeights  = cl::Buffer(m_context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, m_hidWeights.size() *sizeof(float), m_hidWeights.data());
            m_tmp       = cl::Buffer(m_context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, m_hidPerceptrons*sizeof(float), m_hidOutput.data());

            m_calcHiddenOutput.setArg(0, m_eta);
            m_calcHiddenOutput.setArg(1, m_inpPerceptrons);
            m_calcHiddenOutput.setArg(2, m_hidPerceptrons);
            m_calcHiddenOutput.setArg(3, m_hWeights);
            m_calcHiddenOutput.setArg(4, m_trImg);
            m_calcHiddenOutput.setArg(5, m_tmp);
        }
        catch (const cl::Error &err)
        {
            std::cerr << "init opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
            return false;
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool OpenCLPerceptron::initTesting(std::vector<float> *testImg)
{
    if(testImg != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void OpenCLPerceptron::trainAll()
{
    try
    {
        cl::CommandQueue queue(m_context, m_device[0]);
        queue.enqueueNDRangeKernel(m_calcHiddenOutput, cl::NullRange, m_hidPerceptrons, cl::NullRange);
        queue.enqueueReadBuffer(m_tmp, CL_TRUE, 0, m_hidOutput.size() * sizeof(float), m_hidOutput.data());
        std::cout << "values: ";
        for(int i=0; i< m_hidPerceptrons; ++i)
        {
            std::cout << m_hidOutput[i] << " ";
        }
        std::cout << std::endl;
    }
    catch( const cl::Error &err)
    {
        std::cerr << "trainall opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

std::vector<float>* OpenCLPerceptron::testAll()
{
    return nullptr;
}
