
#include <iostream>
#include <sstream>
#include <armadillo>
#include <functional>
#include <Windows.h>
#include "CSVKit.h"
#include <omp.h>
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
constexpr size_t nx = 500, ny = 500;
constexpr double d = 0.2;
constexpr double poissonPrecision = 0.00001;
constexpr size_t sfieldelemsize = nx * ny * sizeof(double);

class scalarfield
{
public:
	vector<double> v;
	scalarfield():v(nx*ny){}
	double& at(size_t x, size_t y)
	{
		return v[x * nx + y];
	}
	double constat(size_t x, size_t y) const
	{
		return v[x * nx + y];
	}
};

void SolvePoisson_Core(const scalarfield& f, scalarfield& u, function<double(long long, long long,const scalarfield & u)> boundary)
{
	scalarfield u1, u2;
	scalarfield* uk=&u1, * ukp1=&u2;
	memset(u1.v.data(), 0, sfieldelemsize);
	memset(u2.v.data(), 0, sfieldelemsize);
	auto getu= [boundary, uk](long long x, long long y)->double
	{
		if ((x == -1) || (x == nx) || (y == -1) || (y == ny))
			return boundary(x, y,*uk);
		else
			return (*uk).at(x,y);
	};
	while (true)
	{
		double delta = 0;
#pragma omp parallel for reduction(+:delta)
		for (long long x = 0; x < nx; x++)
		{
			for (long long y = 0; y < ny; y++)
			{
				(*ukp1).at(x,y) =
					(getu(x, y - 1) + getu(x, y + 1) + getu(x - 1, y) + getu(x + 1, y) - d * d * f.constat(x,y)) / 4;
				delta += pow((*ukp1).at(x,y) - (*uk).at(x,y), 2);
			}
		}
		delta /= nx * ny;
		if (delta < poissonPrecision)
		{
			memcpy(u.v.data(), (*ukp1).v.data(), nx * ny * sizeof(double));

			return;
		}
		else
		{
			auto tmp = uk;
			uk = ukp1;
			ukp1 = tmp;
		}
	}
}


void SolvePoisson_Mix(const scalarfield& f, scalarfield& u, function<double(long long, long long,bool& D)> boundary)
{//set D to be true in boundary to express return value is Dirichlet boundary condition
	auto boundary_Dirichlet = [boundary](long long x, long long y, const scalarfield& u)
	{
		bool D = false;
		double valbound = boundary(x, y,D);
		if (D)
			return valbound;
		else
		{
			if (x == -1)
			{
				x = 0;
				return u.constat(x, y) - d * valbound;
			}
			if (x == nx)
			{
				x = nx - 1;
				return u.constat(x, y) + d * valbound;
			}
			if (y == -1)
			{
				y = 0;
				return u.constat(x, y) - d * valbound;
			}
			if (y == ny)
			{
				y = ny - 1;
				return u.constat(x, y) + d * valbound;
			}
			return 0.0;//impossible!
		}
	};
	SolvePoisson_Core(f, u, boundary_Dirichlet);
}

void SolvePoisson_Newman(const scalarfield& f, scalarfield& u, function<double(long long, long long)> boundary)
{
	auto boundary_Dirichlet = [boundary](long long x, long long y, const scalarfield& u)
	{
		double valbound = boundary(x, y);
		if (x == -1)
		{
			x = 0;
			return u.constat(x, y) - d * valbound;
		}
		if (x == nx)
		{
			x = nx - 1;
			return u.constat(x, y) + d * valbound;
		}
		if (y == -1)
		{
			y = 0;
			return u.constat(x, y) - d * valbound;
		}
		if (y == ny)
		{
			y = ny - 1;
			return u.constat(x, y) + d * valbound;
		}
		return 0.0;
	};
	SolvePoisson_Core(f, u, boundary_Dirichlet);
}

void SolvePoisson_Dirichlet(const scalarfield& f, scalarfield& u, function<double(long long, long long)> boundary)
{
	auto boundary_Dirichlet = [boundary](long long x, long long y, const scalarfield& u)
	{
		return boundary(x,y);
	};
	SolvePoisson_Core(f, u, boundary_Dirichlet);
}

inline void partialX(const scalarfield& u,scalarfield& ux, function<double(long long, long long)> ub)
{
	auto getu = [ub, u](long long x, long long y)->double
	{
		if ((x == -1) || (x == nx) || (y == -1) || (y == ny))
			return ub(x, y);
		else
			return u.constat(x, y);
	};

#pragma omp parallel for
	for (long long x = 0; x < nx; x++)
	{
		for (long long y = 0; y < ny; y++)
		{
			ux.at(x,y) = (getu(x + 1, y) - getu(x - 1, y)) / (2 * d);
		}
	}
}

inline void partialY(const scalarfield& u, scalarfield& uy, function<double(long long, long long)> ub)
{
	auto getu = [ub, u](long long x, long long y)->double
	{
		if ((x == -1) || (x == nx) || (y == -1) || (y == ny))
			return ub(x, y);
		else
			return u.constat(x,y);
	};

#pragma omp parallel for
	for (long long x = 0; x < nx; x++)
	{
		for (long long y = 0; y < ny; y++)
		{
			uy.at(x,y) = (getu(x, y + 1) - getu(x, y - 1)) / (2 * d);
		}
	}
}

inline void grad(const scalarfield& phi, scalarfield& u, scalarfield& v, function<double(long long, long long)> phib)
{
	partialX(phi, u, phib);
	partialY(phi, v, phib);
}

void savesfield(scalarfield& P, const string& path)
{
	CSV2D csv;
	for (long long y = 0; y < ny; y++)
	{
		CSVLine line;
		for (long long x = 0; x < nx; x++)
		{
			appendDataPoint2D(line, P.at(x,y));
		}
		appendDataLine2D(csv, line, 0, true);
	}
	saveCSV(csv, path);
}

void PoiTest_Edge()
{
	scalarfield P;
	scalarfield f;
	memset(f.v.data(), 0, f.v.size() * sizeof(double));
	auto t0 = omp_get_wtime();
	SolvePoisson_Dirichlet(f, P, [](long long x, long long y) {return 1000; });
	cout << omp_get_wtime() - t0 << "s" << endl;

	savesfield(P, R"(D:\poisson.csv)");
}

void PoiTest_Cent()
{
	scalarfield P;
	scalarfield f;
	memset(f.v.data(), 0, f.v.size() * sizeof(double));
	constexpr long long size = 10;
	for (long long x = -size; x < size; x++)
	{
		for (long long y = -size; y < size; y++)
		{
			f.at(nx / 2 + x, ny / 2 + y) = -10000;
		}
	}
	auto t0 = omp_get_wtime();
	SolvePoisson_Dirichlet(f, P, [](long long x, long long y) {return 0; });
	cout << omp_get_wtime() - t0 << "s" << endl;
	savesfield(P, R"(D:\poisson.csv)");
}

void PoiTest_EdgeLeftRight()
{
	scalarfield P;
	scalarfield f;
	memset(f.v.data(), 0, f.v.size() * sizeof(double));
	auto Pb = [](long long x, long long y, bool& D)//D is 'Dirichlet'
	{
		//if (x == -1 && y > 40 && y < 60)
		//	return -1000.0;
		//else if (x == nx && y > ny / 2 - 10 && y < ny / 2 + 10)
		//	return 1000.0;
		D = true;
		if (x == -1)
		//if (x == -1 && y <= ny * 3 / 5 && y >= ny * 2 / 5)
			return -1000.0;
		else if (x == nx)
			return 1000.0;
		else
		{
			D = false;
			return 0.0;
		}
	};
	auto t0 = omp_get_wtime();
	SolvePoisson_Mix(f, P, Pb);
	cout << omp_get_wtime() - t0 << "s" << endl;

	constexpr double bottleR2 = 400;
#pragma omp parallel for
	for (long long x = 0; x < nx; x++)
	{
		for (long long y = 0; y < ny; y++)
		{
			double X = d*(x - double(nx / 2)), Y = d*(y - double(ny / 2));
			double R2 = pow(X, 2) + pow(Y, 2);
			if (R2 < bottleR2)
			{
				P.at(x, y) = 0;
			}
			else
			{
				P.at(x, y) += 40000 * (X/pow(R2,1.5));
			}
		}
	}

	savesfield(P, R"(D:\poisson.csv)");

	scalarfield gradP1, gradP2;
	auto Pb1=[P](long long x, long long y)
	{
		if (x == -1)
			return -1000.0;
		else if (x == nx)
			return 1000.0;
		else
			return P.constat(x, y);
	};
	grad(P, gradP1, gradP2, Pb1);
	

//#pragma omp parallel for
//	for (long long x = 0; x < nx; x++)
//	{
//		for (long long y = 0; y < ny; y++)
//		{
//			double X = (x - double(nx / 2)), Y = (y - double(ny / 2));
//			double R2 = pow(X, 2) + pow(Y, 2);
//			if (R2 < bottleR2)
//			{
//				gradP1.at(x, y) = 0;
//				gradP2.at(x, y) = 0;
//			}
//			else
//			{
//				gradP1.at(x, y) -= 5000 * (3 * X * X / pow(R2, 2.5) - 1 / pow(R2, 1.5));
//				gradP2.at(x, y) -= 5000 * (3 * X * Y / pow(R2, 2.5));
//			}
//		}
//	}
	savesfield(gradP1, R"(D:\gradP1.csv)");
	savesfield(gradP2, R"(D:\gradP2.csv)");
}

void PoiTest_Dipole()
{

	scalarfield gradP1, gradP2;

	
	constexpr double bottleR2 = 100;
	#pragma omp parallel for
		for (long long x = 0; x < nx; x++)
		{
			for (long long y = 0; y < ny; y++)
			{
				double X = (x - double(nx / 2)), Y = (y - double(ny / 2));
				double R2 = pow(X, 2) + pow(Y, 2);
				if (R2 < bottleR2)
				{
					gradP1.at(x, y) = 0;
					gradP2.at(x, y) = 0;
				}
				else
				{
					gradP1.at(x, y) += 1000*(3*X * X / pow(R2, 2.5)-1/ pow(R2, 1.5));
					gradP2.at(x, y) += 1000 * (3 * X * Y / pow(R2, 2.5));
				}
			}
		}
	savesfield(gradP1, R"(D:\gradP1.csv)");
	savesfield(gradP2, R"(D:\gradP2.csv)");
}

int main()
{
	PoiTest_EdgeLeftRight();
}


