#include <iostream>

#include "mlp.h"

int main()
{
    MLP bp;

    const float p00 [] = {0.0f, 0.0f};
    const float p01 [] = {0.0f, 1.0f};
    const float p10 [] = {1.0f, 0.0f};
    const float p11 [] = {1.0f, 1.0f};

    //train the mlp
    const unsigned long iterations = 5000;
    for ( unsigned  long i = 0; i < iterations; ++i )
    {

        bp.train( p00, 0 );
        bp.train( p01, 1 );
        bp.train( p10, 1 );
        bp.train( p11, 0 );
    }

    //see if it learned something
    std::cout << "0,0 = " << bp.run( p00 ) << std::endl;
    std::cout << "0,1 = " << bp.run( p01 ) << std::endl;
    std::cout << "1,0 = " << bp.run( p10 ) << std::endl;
    std::cout << "1,1 = " << bp.run( p11 ) << std::endl;

    return 0;
}
