float sigmoid( const float pNum )
{
    return (1.0f/(1.0f+exp(-pNum)));
}

/*
    Weights should be stored in rows, not columns
*/

kernel void calcLayer(
    const int preP,
    const int curP,
    global const float *curWeight,
    global const float *curInp,
    const int inpOffset,
    global float *curOutput
    )
{
    size_t i = get_global_id(0);
    if(i < curP)
    {
        //add constant
        curOutput[i] = 1*curWeight[i*preP];
        //sum up all inputs*weightings
        for(int j=0; j<preP; ++j)
        {
            curOutput[i] += curInp[j+inpOffset] * curWeight[j+1+i*preP];
        }
        curOutput[i] = sigmoid(curOutput[i]);
    }
}

kernel void calcLayerDelta(
    const int curP,
    const int nexP,
    global const float *curOutput,
    global const float *nexWeights,
    global const float *nexDelta,
    global float *curDelta
    )
{
    size_t i = get_global_id(0);
    if(i < curP)
    {
        curDelta[i] = curOutput[i]*(1-curOutput[i]);
        float tmpSum = 0;
        for(int j=0; j<nexP; ++j)
        {
            tmpSum += nexWeights[j+1+i*nexP]*nexDelta[j];
        }
        curDelta[i] *= tmpSum;
    }
}

kernel void calcOutputDelta(
    const int outP,
    global const float *outOutput,
    global const float *target,
    const int targetOffset,
    global float *outDelta
    )
{
    size_t i = get_global_id(0);
    if(i < outP)
    {
        outDelta[i] = outOutput[i]*(1-outOutput[i])*(target[i+targetOffset]-outOutput[i]);
    }
}

kernel void applyDelta(
    const int curP,
    const float eta,
    global const float *curDelta,
    global const float *curOutput,
    global float *curWeight
    )
{
    size_t i = get_global_id(0);
    if(i < curP)
    {
        curWeight[i] += eta*1*curDelta[i];
        for(int j=0; j<curP; ++j)
        {
            curWeight[j+1+i*curP] += eta*curOutput[j]*curDelta[i];
        }
    }
}
