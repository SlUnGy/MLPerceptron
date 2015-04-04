#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class OpenCLPerceptron
{
public:
    OpenCLPerceptron();
    ~OpenCLPerceptron();

    bool hasFoundDevice(){return m_foundDevice;}

    bool initOpenCL();
    bool initTraining(const std::vector<float>&, const std::vector<float>&);
    bool initTesting(const std::vector<float>&);

    void trainAll();
    //will only do classification
    float** testAll();
protected:
    bool m_foundDevice;
    const std::string m_sourceFile;
    const std::string m_kernelName;
private:
    cl::Context m_context;
    std::vector<cl::Device> m_device;
};

#endif // OCLP_H_INCLUDED
