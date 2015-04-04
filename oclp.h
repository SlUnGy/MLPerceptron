#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#include <string>
#include <vector>

class OpenCLPerceptron
{
public:
    OpenCLPerceptron();
    ~OpenCLPerceptron();

    bool hasFoundDevice(){return m_foundDevice;}

    int initTraining(const std::vector<float>&, const std::vector<float>&);
    int initTesting(const std::vector<float>&)

    void trainAll();
    //will only do classification
    float** testAll();
protected:
    bool m_foundDevice;
    const std::string m_sourceFile;
    const std::string m_kernelName;
private:
};

#endif // OCLP_H_INCLUDED
