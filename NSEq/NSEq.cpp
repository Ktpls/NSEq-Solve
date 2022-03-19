
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
constexpr size_t nx = 100, ny = 100,nt=2000;
constexpr double d = 1,dt=0.001;
constexpr double poissonPrecision = 0.001;
constexpr size_t sfieldelemsize = nx * ny * sizeof(double);

//using sfield = double[nx][ny];

class sfield
{
public:
	vector<double> v;
	sfield():v(nx*ny){}
	double& at(size_t x, size_t y)
	{
		return v[x * nx + y];
	}
	double constat(size_t x, size_t y) const
	{
		return v[x * nx + y];
	}
};
sfield Pglobal;

void SolvePoisson_Core(const sfield& f, sfield& u, function<double(long long, long long,const sfield & u)> boundary)
{
	sfield u1, u2;
	sfield* uk=&u1, * ukp1=&u2;
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

void SolvePoisson_Newman(const sfield& f, sfield& u, function<double(long long, long long)> boundary)
{
	auto boundary_Dirichlet = [boundary](long long x, long long y, const sfield& u)
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

void SolvePoisson_Dirichlet(const sfield& f, sfield& u, function<double(long long, long long)> boundary)
{
	auto boundary_Dirichlet = [boundary](long long x, long long y, const sfield& u)
	{
		return boundary(x,y);
	};
	SolvePoisson_Core(f, u, boundary_Dirichlet);
}

inline void partialX(const sfield& u,sfield& ux, function<double(long long, long long)> ub)
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

inline void partialY(const sfield& u, sfield& uy, function<double(long long, long long)> ub)
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

inline void grad(const sfield& phi, sfield& u, sfield& v, function<double(long long, long long)> phib)
{
	partialX(phi, u, phib);
	partialY(phi, v, phib);
}

//void StepNS(sfield& u, sfield& v, function<double(long long, long long)> ub, function<double(long long, long long)> vb)
//{
//	sfield gradP1, gradP2;
//	sfield um, vm;
//
//	sfield ux, uy, vx, vy;
//	partialX(u, ux, ub);
//	partialX(v, vx, vb);
//	partialY(u, uy, ub);
//	partialY(v, vy, vb);
//
//	//gradP
//	{
//		sfield P;
//		sfield laplaceP;
//#pragma omp parallel for
//		for (long long x = 0; x < nx; x++)
//		{
//			for (long long y = 0; y < ny; y++)
//			{
//				//laplace(p) = -2 ( du/dy * dv/dx - du/dx * dv/dy)
//				laplaceP.at(x,y) = -2 * (uy.at(x,y) * vx.at(x,y) - ux.at(x,y) * vy.at(x,y));
//			}
//		}
//		SolvePoisson(laplaceP, P,
//			[](long long x, long long y)
//			{
//				if (x == -1 && y < 60 && y>40)
//					return 1000;
//				else
//					return 0;
//			});
//		//#pragma omp parallel for
//		//for (long long x = 0; x < nx; x++)
//		//{
//		//	for (long long y = 0; y < ny; y++)
//		//	{
//		//		P.at(x, y) = 100 - 0.5 * (pow(u.at(x, y), 2) + pow(v.at(x, y), 2));
//		//	}
//		//}
//		memcpy(Pglobal.v.data(), P.v.data(), sfieldelemsize);
//		grad(P, gradP1, gradP2, [](long long x, long long y) {return 0; });
//	}
//
//	//um,vm
//	{
//#pragma omp parallel for
//		for (long long x = 0; x < nx; x++)
//		{
//			for (long long y = 0; y < ny; y++)
//			{
//				um.at(x,y) = u.at(x,y) * ux.at(x,y) + v.at(x,y) * uy.at(x,y);
//				vm.at(x,y) = u.at(x,y) * vx.at(x,y) + v.at(x,y) * vy.at(x,y);
//			}
//		}
//	}
//
//#pragma omp parallel for
//	for (long long x = 0; x < nx; x++)
//	{
//		for (long long y = 0; y < ny; y++)
//		{
//			u.at(x,y) += (-gradP1.at(x,y) - um.at(x,y)) * dt;
//			v.at(x,y) += (-gradP2.at(x,y) - vm.at(x,y)) * dt;;
//		}
//	}
//}

void savesfield(sfield& P, const string& path)
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

//void PoiTest_Edge()
//{
//	sfield P;
//	sfield f;
//	memset(f.v.data(), 0, f.v.size() * sizeof(double));
//	auto t0 = omp_get_wtime();
//	SolvePoisson(f, P, [](long long x, long long y) {return 1000; });
//	cout << omp_get_wtime() - t0 << "s" << endl;
//
//	savesfield(P, R"(D:\poisson.csv)");
//}

void PoiTest_EdgeLeftRight()
{
	sfield P;
	sfield f;
	memset(f.v.data(), 0, f.v.size() * sizeof(double));
	auto Pb = [](long long x, long long y, const sfield& u)
	{
		if (x == -1&&y>40&&y<60)
			return -1000.0;
		else if (x == nx && y > ny/2-10 && y < ny / 2 + 10)
			return 1000.0;
		if (x == -1)
			x = 0;
		if (x == nx)
			x = nx - 1;
		if (y == -1)
			y = 0;
		if (y == ny)
			y = ny - 1;
		return u.constat(x, y);
		//return 1000;
	};
	auto t0 = omp_get_wtime();
	SolvePoisson_Core(f, P, Pb);
	cout << omp_get_wtime() - t0 << "s" << endl;

	savesfield(P, R"(D:\poisson.csv)");

	sfield gradP1, gradP2;
	auto Pb1=[P](long long x, long long y)
	{
		if (x == -1 && y > 40 && y < 60)
			return -1000.0;
		else if (x == nx && y > ny / 2 - 10 && y < ny / 2 + 10)
			return 1000.0;
		else
			return P.constat(x, y);
	};
	grad(P, gradP1, gradP2, Pb1);
	savesfield(gradP1, R"(D:\gradP1.csv)");
	savesfield(gradP2, R"(D:\gradP2.csv)");
}

//void PoiTest_Cent()
//{
//	sfield P;
//	sfield f;
//	memset(f.v.data(), 0, f.v.size() * sizeof(double));
//	constexpr long long size = 10;
//	for (long long x = -size; x < size; x++)
//	{
//		for (long long y = -size; y < size; y++)
//		{
//			f.at(nx / 2 + x,ny / 2 + y) = -10000;
//		}
//	}
//	auto t0 = omp_get_wtime();
//	SolvePoisson(f, P, [](long long x, long long y) {return 0; });
//	cout << omp_get_wtime() - t0 << "s" << endl;
//	savesfield(P, R"(D:\poisson.csv)");
//}


int main2()
{
	//auto ub = [](long long x, long long y)
	//{
	//	if (x == -1)
	//		return 10;
	//	else
	//		return 0;
	//};
	//auto vb = [](long long x, long long y)
	//{
	//	return 0;
	//};	
	auto ub = [](long long x, long long y)
	{
		return 0;
	};
	auto vb = [](long long x, long long y)
	{
		return 0;
	};

	sfield u, v;
	memset(u.v.data(), 0, u.v.size() * sizeof(double));
	for (size_t y = 0; y < ny; y++)
	{
		u.at(0, y) = 10;
	}
	memset(v.v.data(), 0, v.v.size() * sizeof(double));

	constexpr size_t bottleR = 10;

	char buffer[256];
	for (size_t t = 0; t < nt; t++)
	{
		//todo:set bottle edge
		//{
		//	for (size_t x = nx / 2 - bottleR; x < nx / 2 + bottleR; x++)
		//	{
		//		for (size_t y = ny / 2 - bottleR; y < ny / 2 + bottleR; y++)
		//		{
		//			if (pow(nx / 2.0 - x, 2) + pow(ny / 2.0 - y, 2) < bottleR)
		//			{
		//				u.at(x,y) = v.at(x,y) = 0;
		//			}
		//		}
		//	}
		//	bool edged[2 * bottleR][2 * bottleR] = { 0 };
		//	for (double theta = 0; theta < 2 * M_PI; theta += 2 * M_PI / 180)
		//	{
		//		double cost = cos(theta), sint = sin(theta);
		//		if (edged[size_t(cost * bottleR + bottleR + 0.5)][size_t(sint * bottleR + bottleR + 0.5)])
		//			continue;
		//		else
		//		{
		//			arma::vec2 vedge({ -sint * bottleR, cost * bottleR });
		//			arma::vec2 vfluid({
		//				u.at(size_t(nx / 2 + cost * bottleR),ny / 2 + size_t(sint * bottleR)),
		//				v.at(size_t(nx / 2 + cost * bottleR),ny / 2 + size_t(sint * bottleR)) });
		//			vedge *= arma::dot(vedge, vfluid);
		//			u.at(size_t(nx / 2 + cost * bottleR),ny / 2 + size_t(sint * bottleR)) = vedge[0];
		//			v.at(size_t(nx / 2 + cost * bottleR),ny / 2 + size_t(sint * bottleR)) = vedge[1];
		//			edged[size_t(cost * bottleR + bottleR + 0.5)][size_t(sint * bottleR + bottleR + 0.5)] = true;
		//		}
		//	}
		//}

		//StepNS(u, v, ub, vb);
		
		sprintf_s(buffer, "D:/NSEq/u_%ld.csv", t);
		savesfield(u, buffer);

		sprintf_s(buffer, "D:/NSEq/v_%ld.csv", t);
		savesfield(v, buffer);

		sprintf_s(buffer, "D:/NSEq/p_%ld.csv", t);
		savesfield(Pglobal, buffer);
		cout << t << "/" << nt << endl;
	}
	return 0;
}

int main()
{
	PoiTest_EdgeLeftRight();
}