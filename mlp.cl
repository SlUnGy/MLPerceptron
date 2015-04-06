float sigmoid( const float pNum )
{
    return (1.0f/(1.0f+exp(-pNum)));
}

/*
    Weights should be stored in rows, not columns
*/

kernel void calcHidden(
    const float eta,
    const int inpP,
    const int hidP,
    global const float *hidWeight,
    global const float *image,
    global float *hidOutput
    )
{
    size_t i = get_global_id(0);
    if(i < hidP)
    {
        //add constant
        hidOutput[i] = 1*hidWeight[i*inpP];
        //sum up all inputs*weightings
        for(int j=1; j<inpP+1; ++j)
        {
            hidOutput[i] += image[j-1] * hidWeight[j+i*inpP];
        }
        hidOutput[i] = sigmoid(hidOutput[i]);
    }
}

kernel void calcOut()
{
    size_t i = get_global_id(0);
    /*if (i < n)
    {
        c[i] = a[i] + b[i];
    }*/
}

kernel void calcDelta()
{

}

kernel void backprop()
{

}
