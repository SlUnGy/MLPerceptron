#include "loadData.h"
#include "tdata.h"

#include <iostream>

void parseCommandlineParameters(int argc, char* argv[], TrainingEnvironment &pEnv)
{
    if(argc > 1)
    {
        std::vector<std::string> cmdParams(argc);
        for(int i=1; i<argc; ++i)
        {
            cmdParams[i] = std::string(argv[i]);
        }
        for(const std::string &par : cmdParams)
        {
            if(par == "-parallel" || par == "-p")
            {
                pEnv.setType(parallel);
            }
            else if(par == "-sequential" || par == "-s")
            {
                pEnv.setType(sequential);
            }
            else if( par.find_first_not_of(' ') != std::string::npos )//ignore whitespace
            {
                std::cerr << "didn't recognize the option: " << par << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    TrainingEnvironment tEnv;

    parseCommandlineParameters(argc, argv, tEnv);

    std::cout << "loading and initialising images." << std::endl;
    if(loadImageData(tEnv))
    {
        return tEnv.executeOCR();
    }
    else
    {
        return 3;
    }
}
