#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#include <string>

class OpenCLPerceptron
{
public:
    OpenCLPerceptron();
    ~OpenCLPerceptron();

    bool hasFoundDevice(){return m_foundDevice;}
    int init();

    void train(const float**, const float**);
    float** classify(const float**);
protected:
    bool m_foundDevice;
    const std::string m_sourceFile;
    const std::string m_kernelName;
private:
};

#endif // OCLP_H_INCLUDED
