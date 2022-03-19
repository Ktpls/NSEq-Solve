#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

using CSV1D = string;
using CSVLine = string;
using CSV2D = string;
using CSV = string;
using axisvec1D = vector<double>;
using axisvec2D = vector<vector<double>>;
inline void appendDataPoint1D(CSV1D& s, double x, double y)
{
	char buf[256] = { 0 };
	sprintf_s(buf, "%lf,%lf\n", x, y);
	s.append(buf);
}
inline void vec2csv1d(CSV1D& s, const axisvec1D& xs, const axisvec1D& ys)
{
	size_t len = xs.size();
	for (size_t i = 0; i < len; i++)
		appendDataPoint1D(s, xs[i], ys[i]);
}
inline void appendDataPoint2D(CSVLine& s, double x)
{
	char buf[256] = { 0 };
	sprintf_s(buf, "%lf,", x);
	s.append(buf);
}
inline void appendDataLine2D(CSV2D& table, const CSVLine& line, double head,bool nohead=false)
{
	char buf[256] = { 0 };
	if(!nohead)
		sprintf_s(buf, "%lf,", head);
	table.append(buf);
	table.append(line);
	*(rbegin(table)) = '\n';
}
inline void buildHead2D(CSV2D& table, const axisvec1D& head)
{
	CSVLine headline;
	for (auto h : head)
	{
		appendDataPoint2D(headline, h);
	}
	appendDataLine2D(table, headline, 0);

}
inline void saveCSV(const CSV& s, const string& path)
{
	ofstream of(path, ios::binary);
	if (s.size() != 0)
		of.write(s.c_str(), s.size() - 1); //trim \n at the end
	of.close();
}