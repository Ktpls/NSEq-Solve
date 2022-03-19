#pragma once
inline double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}
inline double pcomb_lnr(double a, double b, double pr)
{
    return pr * a + (1 - pr) * b;
}
inline double pcomb_tri(double a, double b, double pr)
{
    pr *= M_PI_2;
    return sin(pr) * a + cos(pr) * b;
}