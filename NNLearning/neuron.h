#pragma once
#include <vector>
#include <cmath>
#include <random>
using std::vector;
class Neuron
{
public:
	vector<Neuron*> in;
	double (*activation)(double) = tanh;

	static std::default_random_engine random_w_engine;
	vector<double> w;
	double threshold = 0;
	double out = 0;
	vector<double> diff_error_in;
	vector<double> diff_error_w;
	double diff_error_threshold = 0;
	vector<double> delta_w;
	double delta_threshold = 0;

	//static double sgn(double x) { return (x < 0 ? 0 : 1); }
	static double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
	static double tanh(double x) { return 1 - 2 / (std::exp(2 * x) + 1); }

	Neuron() {}
	//Neuron(std::initializer_list<Neuron*> il) : in(il) { init_w(); }

	void init_w()
	{
		static std::uniform_real_distribution<double> d(-1, 1);
		w.clear();
		for (int i = 0; i < in.size(); i++)
			w.push_back(d(random_w_engine));
		threshold = d(random_w_engine);

		diff_error_in = vector<double>(in.size(), 0);
		diff_error_w = vector<double>(in.size(), 0);
		delta_w = vector<double>(in.size(), 0);
		delta_threshold = 0;
	}
	void update_out() // update out
	{
		double x = 0;
		for (size_t i = 0; i < in.size(); i++)
			x += in[i]->out * w[i];
		x -= threshold;
		out = activation(x);
	}
	void update_diff(double diff_error_out) // update diff_*
	{
		double diff_error_x = 0;
		if (activation == sigmoid)
		{
			double sigmoid_diff = out * (1 - out);
			diff_error_x = diff_error_out * sigmoid_diff;
		}
		else if (activation == tanh)
		{
			double tanh_diff = 1 - out * out;
			diff_error_x = diff_error_out * tanh_diff;
		}
		diff_error_threshold = -diff_error_x;
		for (size_t i = 0; i < in.size(); i++)
		{
			diff_error_w[i] = diff_error_x * in[i]->out;
			diff_error_in[i] = diff_error_x * w[i];
		}
	}
	void bp(double learning_rate) // update delta_w and delta_threshold
	{
		delta_threshold += -learning_rate * diff_error_threshold;
		for (size_t i = 0; i < in.size(); i++)
			delta_w[i] += -learning_rate * diff_error_w[i];
	}
	void apply_new_w() // update w and threshold
	{
		threshold += delta_threshold;
		delta_threshold = 0;
		for (size_t i = 0; i < in.size(); i++)
			w[i] += delta_w[i];
		delta_w = vector<double>(in.size(), 0);
	}
	void apply_new_w(double rate) // update w and threshold
	{
		threshold += delta_threshold * rate;
		delta_threshold = 0;
		for (size_t i = 0; i < in.size(); i++)
			w[i] += delta_w[i] * rate;
		delta_w = vector<double>(in.size(), 0);
	}
};
std::default_random_engine Neuron::random_w_engine(68441468);