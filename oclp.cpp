#include "oclp.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <random>

#include <algorithm>

OpenCLPerceptron::OpenCLPerceptron(const float pEta, const int pInputPerceptrons, const int pHiddenPerceptrons, const int pOutputPerceptrons)
    :m_foundDevice{false}, m_sourceFile{"mlp.cl"}, m_eta{pEta},
     m_inpPerceptrons{pInputPerceptrons}, m_hidPerceptrons{pHiddenPerceptrons},
     m_outPerceptrons{pOutputPerceptrons},
     m_hidWeights(m_hidPerceptrons*(m_inpPerceptrons+1)),
     m_outWeights(m_outPerceptrons*(m_hidPerceptrons+1)),
     m_trainingDataSets{0}, m_testDataSets{0}
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

    for (unsigned int i=0;i<m_hidWeights.size();i++)
    {
		m_hidWeights[i] = distribution(mt);
    }

    for (unsigned int i=0;i<m_outWeights.size();i++)
    {
		m_outWeights[i] = distribution(mt);
	}
}

bool OpenCLPerceptron::initOpenCL()
{
    try
    {
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
            try
            {
                currentPlatform->getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
            }
            catch (const cl::Error &err)
            {
                std::cerr << "no cpu found, trying next" << std::endl;
            }

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

        std::cout << "using: " << m_device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

        std::ifstream file(m_sourceFile);
        std::string prog(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));

        m_program = cl::Program(m_context, cl::Program::Sources( 1, std::make_pair(prog.c_str(), prog.length()+1)));

        try {
            m_program.build(m_device);
        }
        catch (const cl::Error&)
        {
            std::cerr << "compilation error: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device[0]) << std::endl;
            return false;
        }

        m_calcLayerOutput   = cl::Kernel(m_program, "calcLayer");
        m_calcLayerDelta    = cl::Kernel(m_program, "calcLayerDelta");
        m_calcOutputDelta   = cl::Kernel(m_program, "calcOutputDelta");
        m_applyDelta        = cl::Kernel(m_program, "applyDelta");
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return false;
    }
    return true;
}

bool OpenCLPerceptron::initTraining(std::vector<float> *trainImg, std::vector<float> *trainClf, std::vector<float> *testImg)
{
    if(trainImg != nullptr && trainClf != nullptr && testImg != nullptr)
    {
        try {
            m_bTrImg     = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, trainImg->size()*sizeof(float), trainImg->data());
            m_bTrClf     = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, trainClf->size()*sizeof(float), trainClf->data());
            m_bHWeights  = cl::Buffer(m_context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, m_hidWeights.size()*sizeof(float), m_hidWeights.data());
            m_bHOut      = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_hidPerceptrons*sizeof(float));
            m_bHDelta    = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_hidPerceptrons*sizeof(float));
            m_bOWeights  = cl::Buffer(m_context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, m_outWeights.size()*sizeof(float), m_outWeights.data());
            m_bOOut      = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_outPerceptrons*sizeof(float));
            m_bODelta    = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_outPerceptrons*sizeof(float));

            m_bTeImg     = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, testImg->size()*sizeof(float), testImg->data());

            m_trainingDataSets  = trainImg->size()/m_inpPerceptrons;
            m_testDataSets      = testImg->size()/m_inpPerceptrons;

            const int classificationDataSets = trainClf->size()/m_outPerceptrons;


            if(m_trainingDataSets != classificationDataSets)
            {
                std::cerr << "training data set amount (" << m_trainingDataSets <<
                 ") and training classification set amount(" << classificationDataSets << ") don't match " << std::endl;
                 m_trainingDataSets = 0;
                 return false;
            }
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

void OpenCLPerceptron::trainAll()
{
    try
    {
        cl::CommandQueue queue(m_context, m_device[0]);

        for(int i = 0; i < m_trainingDataSets; ++i)
        {
            const unsigned int imageOffset = i * m_inpPerceptrons;
            const unsigned int classOffset = i * m_outPerceptrons;

            m_calcLayerOutput.setArg(0, m_inpPerceptrons);
            m_calcLayerOutput.setArg(1, m_hidPerceptrons);
            m_calcLayerOutput.setArg(2, m_bHWeights);
            m_calcLayerOutput.setArg(3, m_bTrImg);
            m_calcLayerOutput.setArg(4, imageOffset);
            m_calcLayerOutput.setArg(5, m_bHOut);
            queue.enqueueNDRangeKernel(m_calcLayerOutput, cl::NullRange, m_hidPerceptrons);

            m_calcLayerOutput.setArg(0, m_hidPerceptrons);
            m_calcLayerOutput.setArg(1, m_outPerceptrons);
            m_calcLayerOutput.setArg(2, m_bOWeights);
            m_calcLayerOutput.setArg(3, m_bHOut);
            m_calcLayerOutput.setArg(4, 0);
            m_calcLayerOutput.setArg(5, m_bOOut);
            queue.enqueueNDRangeKernel(m_calcLayerOutput, cl::NullRange, m_outPerceptrons);

            m_calcOutputDelta.setArg(0, m_outPerceptrons);
            m_calcOutputDelta.setArg(1, m_bOOut);
            m_calcOutputDelta.setArg(2, m_bTrClf);
            m_calcOutputDelta.setArg(3, classOffset);
            m_calcOutputDelta.setArg(4, m_bODelta);
            queue.enqueueNDRangeKernel(m_calcOutputDelta, cl::NullRange, m_outPerceptrons);

            m_calcLayerDelta.setArg(0, m_hidPerceptrons);
            m_calcLayerDelta.setArg(1, m_outPerceptrons);
            m_calcLayerDelta.setArg(2, m_bHOut);
            m_calcLayerDelta.setArg(3, m_bOWeights);
            m_calcLayerDelta.setArg(4, m_bODelta);
            m_calcLayerDelta.setArg(5, m_bHDelta);
            queue.enqueueNDRangeKernel(m_calcLayerDelta, cl::NullRange, m_hidPerceptrons);

            m_applyDelta.setArg(0, m_hidPerceptrons);
            m_applyDelta.setArg(1, m_outPerceptrons);
            m_applyDelta.setArg(2, m_eta);
            m_applyDelta.setArg(3, m_bODelta);
            m_applyDelta.setArg(4, m_bHOut);
            m_applyDelta.setArg(5, 0);
            m_applyDelta.setArg(6, m_bOWeights);
            queue.enqueueNDRangeKernel(m_applyDelta, cl::NullRange, m_outPerceptrons);

            m_applyDelta.setArg(0, m_inpPerceptrons);
            m_applyDelta.setArg(1, m_hidPerceptrons);
            m_applyDelta.setArg(2, m_eta);
            m_applyDelta.setArg(3, m_bHDelta);
            m_applyDelta.setArg(4, m_bTrImg);
            m_applyDelta.setArg(5, imageOffset);
            m_applyDelta.setArg(6, m_bHWeights);
            queue.enqueueNDRangeKernel(m_applyDelta, cl::NullRange, m_hidPerceptrons);
        }
    }
    catch( const cl::Error &err)
    {
        std::cerr << "trainall opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

void OpenCLPerceptron::testAll(float *pOutputBuffer)
{
    try
    {
        cl::CommandQueue queue(m_context, m_device[0]);
        for(int i=0; i<m_testDataSets; ++i)
        {
            const int imageOffset = i * m_inpPerceptrons;
            const int outOffset   = i * m_outPerceptrons;

            m_calcLayerOutput.setArg(0, m_inpPerceptrons);
            m_calcLayerOutput.setArg(1, m_hidPerceptrons);
            m_calcLayerOutput.setArg(2, m_bHWeights);
            m_calcLayerOutput.setArg(3, m_bTeImg);
            m_calcLayerOutput.setArg(4, imageOffset);
            m_calcLayerOutput.setArg(5, m_bHOut);
            queue.enqueueNDRangeKernel(m_calcLayerOutput, cl::NullRange, m_hidPerceptrons);

            m_calcLayerOutput.setArg(0, m_hidPerceptrons);
            m_calcLayerOutput.setArg(1, m_outPerceptrons);
            m_calcLayerOutput.setArg(2, m_bOWeights);
            m_calcLayerOutput.setArg(3, m_bHOut);
            m_calcLayerOutput.setArg(4, 0);
            m_calcLayerOutput.setArg(5, m_bOOut);
            queue.enqueueNDRangeKernel(m_calcLayerOutput, cl::NullRange, m_outPerceptrons);

            queue.enqueueReadBuffer(m_bOOut, CL_TRUE, 0, m_outPerceptrons * sizeof(float), pOutputBuffer+outOffset);
        }
    }
    catch( const cl::Error &err )
    {
        std::cerr << "testall opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}
