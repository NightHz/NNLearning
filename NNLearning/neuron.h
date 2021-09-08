#pragma once
#include <vector>
#include <cmath>
using std::vector;
class Neuron
{
public:
	vector<Neuron*> in;
	double (*acti_func)(double) = sigmoid;

	vector<double> w;
	double threshold = 1;
	double out = 0;
	vector<double> diff_error_in;
	vector<double> diff_error_w;
	double diff_error_threshold = 0;
	vector<double> delta_w;
	double delta_threshold = 0;

	//static double sgn(double x) { return (x < 0 ? 0 : 1); }
	static double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

	Neuron() {}
	Neuron(std::initializer_list<Neuron*> il) : in(il) { init_w(); }
	Neuron(std::initializer_list<Neuron*> il, double _w) : in(il) { init_w(_w); }

	void init_w(double _w = 0)
	{
		w.clear();
		w.insert(w.begin(), in.size(), _w);
		diff_error_in.clear();
		diff_error_in.insert(diff_error_in.begin(), in.size(), 0);
		diff_error_w.clear();
		diff_error_w.insert(diff_error_w.begin(), in.size(), 0);
		delta_w.clear();
		delta_w.insert(delta_w.begin(), in.size(), _w);
	}
	void update_out()
	{
		double x = 0;
		for (size_t i = 0; i < in.size(); i++)
			x += in[i]->out * w[i];
		x -= threshold;
		out = acti_func(x);
	}
	void update_diff(double diff_error_out)
	{
		double sigmoid_diff = out * (1 - out);
		double diff_error_x = diff_error_out * sigmoid_diff;
		diff_error_threshold = -diff_error_x;
		for (size_t i = 0; i < in.size(); i++)
		{
			diff_error_w[i] = diff_error_x * in[i]->out;
			diff_error_in[i] = diff_error_x * w[i];
		}
	}
	void bp(double learning_velocity)
	{
		threshold -= learning_velocity * diff_error_threshold;
		for (size_t i = 0; i < in.size(); i++)
			w[i] -= learning_velocity * diff_error_w[i];
	}
};