#include "idxfile.h"

#include <iostream>
#include <fstream>

// read a 32-bit big-endian signed integer
uint32_t read32_be(std::istream& pIn)
{
    char b[4];
    pIn.read(b,4);
    return static_cast<int32_t>((b[3])|(b[2]<<8)|(b[1]<<16)|(b[0]<<24));
}

IDXFile::IDXFile(const std::string &pFile, bool pMSB)
{
    std::ifstream file(pFile, std::ios::in | std::ios::binary);
    if(file.is_open())
    {
        m_magicNumber = read32_be(file);
        const uint8_t firstByte = ( m_magicNumber >> (8*(sizeof(uint32_t)-1))) & 0xFF;
        const uint8_t secondByte = ( m_magicNumber >> (8*(sizeof(uint32_t)-2))) & 0xFF;
        if(firstByte == 0 && secondByte == 0)
        {
            const uint8_t thirdByte = ( m_magicNumber >> (8*(sizeof(uint32_t)-3))) & 0xFF;

            //this should be able to use other datatypes beside 0x08->unsigned byte
            bool recognized = true;
            switch(thirdByte)
            {
            case(0x08):
                break;
            case(0x09)://int8_t
            case(0x0B)://int16_t
            case(0x0C)://int32_t
            case(0x0D)://float
            case(0x0E)://double
            default:
                std::cerr << pFile << ":magic number(" << m_magicNumber << "): unrecognized third byte: " << std::endl;
                recognized = false;
            }
            if(recognized)
            {
                m_dimensionNumber = ( m_magicNumber >> (8*(sizeof(uint32_t)-4))) & 0xFF;
                std::cout << (int) m_dimensionNumber << std::endl;
                m_dimension = new uint32_t[m_dimensionNumber]();
                for(uint8_t i=0; i<m_dimensionNumber; ++i)
                {
                    m_dimension[i] = read32_be(file);
                }
            }
        }
        else
        {
            std::cerr << pFile << ":magic number("<< m_magicNumber <<"): missing two zero bytes: " << (int)firstByte << "/" << (int)secondByte << std::endl;
        }
    }
    else
    {
        std::cerr << pFile << ":couldn't be opened" << std::endl;
    }
}

IDXFile::~IDXFile()
{
    delete [] m_dimension;
    delete m_data;
}
