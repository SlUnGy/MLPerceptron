#include "idxfile.h"

#include <iostream>
#include <fstream>

//read a 32bit bigendian number from an istream
uint32_t read32_be(std::istream& pIn)
{
    char b[4];
    pIn.read(b,4);
    const uint32_t tmp = static_cast<uint32_t>((b[3])|((b[2]<<8)&0x0000ff00)|((b[1]<<16)&0x00ff0000)|((b[0]<<24)&0xff000000));
    return tmp;
}

IDXFile::IDXFile(const std::string& pFile)
    :m_error{false}, m_magicNumber{0}, m_dimensionNumber{0}, m_dimension{nullptr}, m_data{nullptr}
{
    readFile(pFile);
}

IDXFile::IDXFile( const IDXFile& pCpy )
    :m_error{pCpy.hasError()},m_magicNumber{pCpy.getMagicNumber()},m_dimensionNumber{pCpy.getDimensionNumber()},
    m_dimension{pCpy.getDimensions()},m_data{pCpy.getDataPointer()}
{

}

bool IDXFile::readFile(const std::string& pFile)
{
    std::ifstream file(pFile, std::ios::in | std::ios::binary);
    if(file.is_open())
    {
        m_magicNumber = read32_be(file);
        if( file )
        {
            const uint8_t firstByte = ( m_magicNumber >> ( 8 * ( sizeof( uint32_t ) - 1 ) ) ) & 0xFF;
            const uint8_t secondByte = ( m_magicNumber >> ( 8 * ( sizeof( uint32_t ) - 2 ) ) ) & 0xFF;
            if( firstByte == 0 && secondByte == 0 )
            {
                const uint8_t thirdByte = ( m_magicNumber >> ( 8 * ( sizeof( uint32_t ) - 3 ) ) ) & 0xFF;

                //this should be able to use other datatypes beside 0x08->unsigned byte
                bool recognized = true;
                switch( thirdByte )
                {
                case( 0x08 ):
                    break;
                case( 0x09 ): //int8_t
                case( 0x0B ): //int16_t
                case( 0x0C ): //int32_t
                case( 0x0D ): //float
                case( 0x0E ): //double
                default:
                    recognized = false;
                }
                if( recognized )
                {
                    m_dimensionNumber = ( m_magicNumber >> ( 8 * ( sizeof( uint32_t ) - 4 ) ) ) & 0xFF;
                    m_dimension = new uint32_t[m_dimensionNumber]();

                    uint32_t totalSize = 1;
                    for( uint8_t i = 0; i < m_dimensionNumber; ++i )
                    {
                        m_dimension[i] = read32_be( file );
                        if( file )
                        {
                            totalSize *= m_dimension[i];
                        }
                        else
                        {
                            std::cerr << pFile << ":error while reading dimension " << ( int ) i << std::endl;
                            m_error = true;
                            return false;
                        }
                    }
                    m_data = new uint8_t[totalSize]();
                    file.read((char*)m_data, totalSize );
                    if( file )
                    {
                        file.close();
                    }
                    else
                    {
                        std::cerr << pFile << ":error while reading data" << std::endl;
                        m_error = true;
                        return false;
                    }
                }
                else
                {
                    std::cerr << pFile << ":magic number(" << m_magicNumber << "): unrecognized third byte: " << std::endl;
                    m_error = true;
                    return false;
                }
            }
            else
            {
                std::cerr << pFile << ":magic number(" << m_magicNumber << "): missing two zero bytes: " << ( int )firstByte << "/" << ( int )secondByte << std::endl;
                m_error = true;
                return false;
            }
        }
        else
        {
            std::cerr << pFile << ":error while reading magic number " << std::endl;
            m_error = true;
            return false;
        }
    }
    else
    {
        std::cerr << pFile << ":couldn't be opened" << std::endl;
        m_error = true;
        return false;
    }
    return true;
}

IDXFile::~IDXFile()
{
    deleteData();
}
