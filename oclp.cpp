#include "oclp.h"

#include <iostream>
#include <fstream>

OpenCLPerceptron::OpenCLPerceptron()
:m_foundDevice{false},m_sourceFile{"mlp.cl"}
{

}

OpenCLPerceptron::~OpenCLPerceptron()
{

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
            currentPlatform->getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

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

        m_classify          = cl::Kernel(m_program, "classify");
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

bool OpenCLPerceptron::initTraining(const std::vector<float> &trainImg, const std::vector<float> &trainClf)
{
    try {
        const size_t N = 1 << 20;
        std::vector<double> a(N, 1);
        std::vector<double> b(N, 41);

        // Allocate device buffers and transfer input data to device.
        m_trImg = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(double), a.data());
        m_trClf = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(double), b.data());
        C = cl::Buffer(m_context, CL_MEM_READ_WRITE, N * sizeof(double));

        // Set kernel parameters.
        m_classify.setArg(0, static_cast<cl_ulong>(N));
        m_classify.setArg(1, m_trImg);
        m_classify.setArg(2, m_trClf);
        m_classify.setArg(3, C);
    }
    catch (const cl::Error &err)
    {
        std::cerr << "init opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return false;
    }
    return true;
}

bool OpenCLPerceptron::initTesting(const std::vector<float> &testImg)
{
    return true;
}

void OpenCLPerceptron::trainAll()
{
    try
    {
        const size_t N = 1 << 20;
        std::vector<double> c(N);

        cl::CommandQueue queue(m_context, m_device[0]);
        queue.enqueueNDRangeKernel(m_classify, cl::NullRange, N, cl::NullRange);
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
        std::cout << c[42] << std::endl;
    }
    catch( const cl::Error &err)
    {
        std::cerr << "trainall opencl error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

float** OpenCLPerceptron::testAll()
{
    return nullptr;
}
