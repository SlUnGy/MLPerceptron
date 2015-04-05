float sigmoid( const float pNum )
{
    return (1.0f/(1.0f+expf(-pNum)));
}

kernel void calcHidden(
    global const float eta,
    global const int inpP,
    global const int hidP,
    global const float *hidWeight,
    global const float *image,
    global float *hidOutput
    )
{
    size_t i = get_global_id(0);
    if(i < hidP)
    {
        //add constant
        hidOutput[i] = 1;//*hidWeight[0][i];
        //sum up all inputs*weightings
        for(int j=1; j<inpP+1; ++j)
        {
            hidOutput[i] += image[j-1] * 1;//hidWeight[j][i];
        }
        hidOutput[i] = sigmoid(hidOutput[i]);
    }
}

kernel void calcOut(
    global const float eta,
    global const int hidP,
    global const int outP,
    global const float *hidOut,
    global float *output
    )
{
    size_t i = get_global_id(0);
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

kernel void calcDelta()
{

}

kernel void backprop()
{

}
