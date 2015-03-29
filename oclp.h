#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class OpenCLPerceptron
{
public:
    OpenCLPerceptron(cl::Context&,cl::Device&);
    ~OpenCLPerceptron();
protected:
private:
    const std::string sourceFile;
    cl::Program compiledOCL;
};

#endif // OCLP_H_INCLUDED
