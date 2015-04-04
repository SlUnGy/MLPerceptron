kernel void add(
    ulong n,
    global const double *a,
    global const double *b,
    global double *c )
{
    size_t i = get_global_id(0);
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}
