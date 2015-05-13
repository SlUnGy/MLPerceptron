float sigmoid( const float pNum )
{
    return (1.0f/(1.0f+exp(-pNum)));//maybe switch to native_exp for performance gain?
}

/*
    Weights should be stored in rows, not columns
*/


/*
    float hidOutput[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //add constant
        hidOutput[i] = 1*m_hidWeights[0][i];
        //sum up all inputs*weightings
        for(int j=1; j<m_inpPerceptrons+1; ++j)
        {
            hidOutput[i] += pIn[j-1] * m_hidWeights[j][i];
        }
        hidOutput[i] = sigmoid(hidOutput[i]);
    }
    --> (still column based weights)
    for(int i=0; i<curP; ++i)
    {
        curOutput[i] = 1*curWeights[0][i];
        for(int j=0; j<preP; ++j)
        {
            curOutput[i] += curInp[j] * curWeights[j+1][i];
        }
        curOutput[i] = sigmoid(hidOutput[i]);
    }
*/

kernel void calcLayer(
    const int preP,
    const int curP,
    global const float *curWeight,
    global const float *curInp,
    const unsigned int inpOffset,
    global float *curOutput
    )
{
    const size_t i = get_global_id(0);
    if(i < curP)
    {
        //offset for weights
        const int weightOffset = i*(preP+1);

        //add constant
        curOutput[i] = 1*curWeight[weightOffset];

        //sum up all inputs*weightings
        for(int j=0; j<preP; ++j)
        {
            curOutput[i] += curInp[j+inpOffset] * curWeight[j+1+weightOffset];
        }
        curOutput[i] = sigmoid(curOutput[i]);
    }
}

/*
    float hidDelta[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //m_outWeights[i+1][] -> skip the constant coeffecient
        hidDelta[i] = hidOutput[i]*(1-hidOutput[i]);
        float tmpSum = 0;
        for(int j=0; j<m_outPerceptrons; ++j)
        {
            tmpSum += m_outWeights[i+1][j]*outDelta[j];
        }
        hidDelta[i] *= tmpSum;
    }
    -->
    for(int i=0; i<curP; ++i)
    {
        curDelta[i] = curOutput[i]*(1-curOutput[i]);
        float tmpSum = 0;
        for(int j=0; j<nexP; ++j)
        {
            tmpSum += nexWeights[i+1][j]*nexDelta[j]
        }
        nextDelta[j] *= tmpSum;
    }
*/

kernel void calcLayerDelta(
    const int curP,
    const int nexP,
    global const float *curOutput,
    global const float *nexWeights,
    global const float *nexDelta,
    global float *curDelta
    )
{
    const size_t i = get_global_id(0);
    if(i < curP)
    {
        curDelta[i] = curOutput[i]*(1-curOutput[i]);

        float tmpSum = 0;
        for(int j=0; j<nexP; ++j)
        {
            const int weightOffset = j*(curP+1);
            tmpSum += nexWeights[i+1+weightOffset]*nexDelta[j];
        }
        curDelta[i] *= tmpSum;
    }
}

/*
    float outDelta[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        outDelta[i] = output[i]*(1-output[i])*(pTarget[i]-output[i]);
    }
*/

kernel void calcOutputDelta(
    const int outP,
    global const float *outOutput,
    global const float *target,
    const unsigned int targetOffset,
    global float *outDelta
    )
{
    const size_t i = get_global_id(0);
    if(i < outP)
    {
        outDelta[i] = outOutput[i]*(1-outOutput[i])*(target[i+targetOffset]-outOutput[i]);
    }
}


/*
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        m_outWeights[0][i] += m_eta*1*outDelta[i];
        for(int j=0; j<m_hidPerceptrons; ++j)
        {
            m_outWeights[j+1][i] += m_eta*hidOutput[j]*outDelta[i];
        }
    }
    -->
    for(int i=0; i<curP;; ++i)
    {
        curWeight[0][i] += eta*1*curDelta[i];
        for(int j=0; j<preP; ++j)
        {
            curWeight[j+1][i] += eta*preOutput[j]*curDelta[i];
        }
    }
*/

kernel void applyDelta(
    const int preP,
    const int curP,
    const float eta,
    global const float *curDelta,
    global const float *preOutput,
    const unsigned int preOffset,
    global float *curWeight
    )
{
    const size_t i = get_global_id(0);
    if(i < curP)
    {
        //offset for weights
        const int weightOffset = i*(preP+1);

        curWeight[weightOffset] += eta*1*curDelta[i];
        for(int j=0; j<preP; ++j)
        {
            curWeight[j+1+weightOffset] += eta*preOutput[j+preOffset]*curDelta[i];
        }
    }
}
