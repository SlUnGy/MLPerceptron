#include "tdata.h"

#include "parOCR.h"
#include "seqOCR.h"

TrainingEnvironment::TrainingEnvironment()
    :m_type{invalid}, m_trDat{nullptr}, m_trCls{nullptr},
     m_teDat{nullptr}, m_teCls{nullptr}, m_inputWidth{0}, m_outputWidth{0},
     m_device{nullptr}, m_context{nullptr}, m_opencl{false}
{

}

TrainingEnvironment::~TrainingEnvironment()
{
    m_type = invalid;
    if(m_trDat != nullptr)
    {
        m_trDat->clear();
        delete m_trDat;
        m_trDat = nullptr;
    }
    if(m_trCls != nullptr)
    {
        m_trCls->clear();
        delete m_trCls;
        m_trCls = nullptr;
    }
    if(m_teDat != nullptr)
    {
        m_teDat->clear();
        delete m_teDat;
        m_teDat = nullptr;
    }
    if(m_teCls != nullptr)
    {
        m_teCls->clear();
        delete m_teCls;
        m_teCls = nullptr;
    }
}

bool TrainingEnvironment::initOpenCLEnvironment()
{
    if(!hasOpenCLContext())
    {
        try
        {
            std::cout << "setting up opencl." << std::endl;
            m_device = new std::vector<cl::Device>();
            std::vector<cl::Platform> allPlatforms;
            cl::Platform::get(&allPlatforms);

            if (allPlatforms.empty())
            {
                std::cerr << "OpenCL platforms not found." << std::endl;
                m_opencl = false;
            }
            else
            {//iterate through all platform's devices and check if the devices are usable
                bool foundDevice = false;
                for(auto currentPlatform = allPlatforms.begin();
                    !foundDevice && currentPlatform != allPlatforms.end();
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
                        !foundDevice && currentDevice != allDevices.end();
                        currentDevice++)
                    {
                        if (currentDevice->getInfo<CL_DEVICE_AVAILABLE>())//add other selection criteria here
                        {
                            m_device->push_back(*currentDevice);
                            m_context = new cl::Context(*m_device);
                            foundDevice = true;
                        }
                    }
                }

                if (!foundDevice)
                {
                    std::cerr << "no usable device found." << std::endl;
                    m_opencl = false;
                }
                else
                {
                    std::cout << "using: " << (*m_device)[0].getInfo<CL_DEVICE_NAME>() << std::endl;
                    m_opencl = true;
                }
            }
        }
        catch (const cl::Error &err)
        {
            std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << std::endl;
            m_opencl = false;
        }
    }
    return m_opencl;
}

bool TrainingEnvironment::isValid()
{
    const bool noNullptr        = m_trDat != nullptr && m_trCls != nullptr && m_teDat != nullptr && m_teCls != nullptr;
    const bool sampleWidthValid = m_inputWidth>0 && m_outputWidth > 0;
    return m_type != invalid && noNullptr && sampleWidthValid;
}

int TrainingEnvironment::executeOCR()
{
    int retCode = 0;
    if(isValid())
    {
        switch(m_type)//based on which type of training was wanted we choose the appropiate function
        {
        case parallel:
            if(!hasOpenCLContext())
            {
                initOpenCLEnvironment();
            }
            retCode = parallelOCR(m_trDat, m_trCls, m_teDat, m_teCls, m_inputWidth, m_outputWidth, m_device, m_context);
            break;
        case sequential:
            retCode = sequentialOCR(m_trDat, m_trCls, m_teDat, m_teCls, m_inputWidth, m_outputWidth);
            break;
        case invalid:
        default:
            retCode = -1;
            std::cerr << "training type not recognized. Have you set a command line parameter(-s or -p)?" << std::endl;
            break;
        }
    }
    else
    {
        retCode = -2;
    }

    return retCode;
}
