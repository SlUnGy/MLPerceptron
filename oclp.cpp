#include "oclp.h"

#include <iostream>
#include <fstream>

OpenCLPerceptron::OpenCLPerceptron()
:m_foundDevice{false},m_sourceFile{"mlp.cl"},m_kernelName{"add"}
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
        cl::CommandQueue queue(m_context, m_device[0]);
        std::ifstream file(m_sourceFile);
        std::string prog( std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));

        cl::Program program(m_context, cl::Program::Sources( 1, std::make_pair(prog.c_str(), prog.length()+1)));

        try {
            program.build(m_device);
        }
        catch (const cl::Error&)
        {
            std::cerr << "compilation error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device[0]) << std::endl;
            return false;
        }

        cl::Kernel add(program, m_kernelName.c_str());

        // Prepare input data.
        const size_t N = 1 << 20;
        std::vector<double> a(N, 1);
        std::vector<double> b(N, 41);
        std::vector<double> c(N);

        // Allocate device buffers and transfer input data to device.
        cl::Buffer A(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(double), a.data());
        cl::Buffer B(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(double), b.data());
        cl::Buffer C(m_context, CL_MEM_READ_WRITE, c.size() * sizeof(double));

        // Set kernel parameters.
        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        // Launch kernel on the compute device.
        queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
        std::cout << c[42] << std::endl;
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
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

}

float** OpenCLPerceptron::testAll()
{
    return nullptr;
}
