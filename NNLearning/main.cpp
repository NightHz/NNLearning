#include <iostream>
#include "neural_network.h"

using std::cin;
using std::cout;
using std::endl;

vector<double> encode(double x)
{
	vector<double> vd;
	if (x < 0)
	{
		vd.push_back(1.0);
		x = -x;
	}
	else
		vd.push_back(0.0);
	double b = 5.0;
	for (int i = 0; i < 10; i++)
	{
		if (x >= b)
		{
			vd.push_back(1.0);
			x -= b;
		}
		else
			vd.push_back(0.0);
		b /= 2.0;
	}
	return vd;
}
double decode(vector<double> vd)
{
	double d = 0;
	double b = 5.0;
	for (size_t i = 0; i < 10; i++)
	{
		if (vd[i + 1] >= 0.5)
			d += b;
		b /= 2.0;
	}
	if (vd[0] >= 0.5)
		d = -d;
	return d;
}

int main()
{
	cout << "NN..." << endl;
	NeuralNetwork nn{ 11,50,11 };
	cout << "learning..." << endl;
	for (int i = 0; i < 100; i++) // loop
	{
		double error = 0;
		for (double in = 0; in < 10; in += 0.1) // 100 points
		{
			double out = std::sin(in);
			error += nn.test_error(encode(in), encode(out));
		}
		cout << "current sum error : " << error << endl;
		for (double in = 0; in < 10; in += 0.1) // 100 points
		{
			double out = std::sin(in);
			//cout << "learning " << in << " \t| " << out << endl;
			nn.learning(0.2, encode(in), encode(out));
		}
	}
	cout << "testing..." << endl;
	cout << "test in  \t| out(correct out)" << endl;
	for (double in = 0; in < 10; in += 0.05) // 200 points
	{
		double correct_out = std::sin(in);
		nn.set_in(encode(in));
		nn.update_out();
		vector<double> out = nn.get_out();
		cout << "test " << in << "  \t| " << decode(out) << "(" << correct_out << ")" << endl;
	}

	return 1;
}