#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <functional>
class progress_show
{
private:
    size_t progresstotal;
    double precise;
    double pos2shownext;
public:
    progress_show(size_t _progresstotal, double _precise) :
        progresstotal(_progresstotal),
        precise(_precise),
        pos2shownext(0.0)
    {
        if (1.0 / progresstotal > precise)
            precise = 1.0 / progresstotal;
    }
    inline void tryshow(size_t progressnow)
    {
        if (double(progressnow) / progresstotal >= pos2shownext)
        {
            printf("%d%%\n", int(100.0 * progressnow / progresstotal));
            pos2shownext += precise;
        }
    }
};

inline void saveCSVByName(const CSV& s, const string& name)
{
    saveCSV(s, "D:/DataExchange/CSV/" + name + ".csv");
}



void getfuncsample(
    size_t samplenum,
    axisvec1D& xs, axisvec1D& ys,
    function<double(size_t)> f_x, function<double(size_t)> f_y,
    size_t thread_communicating_interval = 100, double precise = 0.1)
{
    progress_show ps(samplenum, precise);
    size_t workdonetotal = 0;
    xs.resize(samplenum); ys.resize(samplenum);
#pragma omp parallel
    {
        size_t workdone_thisthread = 0;
#pragma omp for
        for (long long i = 0; i < samplenum; i++)
        {
            xs[i] = f_x(i);
            ys[i] = f_y(i);

            workdone_thisthread++;
            if (workdone_thisthread % thread_communicating_interval == 0)
#pragma omp critical
            {
                workdonetotal += thread_communicating_interval;
                ps.tryshow(workdonetotal);
            }
        }
    }
}

void getfuncsample_nonpara(
    size_t samplenum,
    axisvec1D& xs, axisvec1D& ys,
    function<double(size_t)> f_x, function<double(size_t)> f_y,
    size_t thread_communicating_interval = 100, double precise = 0.1)
{
    xs.resize(samplenum); ys.resize(samplenum);
    for (long long i = 0; i < samplenum; i++)
    {
        xs[i] = f_x(i);
        ys[i] = f_y(i);
    }
}

void getarealfunc(
    double xbeg, double xend, size_t samplenum,
    axisvec1D& xs, axisvec1D& ys,
    function<double(double)> f_Y,
    size_t thread_commu_interval = 100, double precise = 0.1)
{
    double xstep = (xend - xbeg) / samplenum;
    auto _fx = [xbeg, xstep](size_t i) { return xbeg + i * xstep; };
    auto _fy = [f_Y, xbeg, xstep](size_t i) { return f_Y(xbeg + i * xstep); };
    getfuncsample(samplenum, xs, ys, _fx, _fy);
}