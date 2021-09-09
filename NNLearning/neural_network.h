#pragma once
#include "neural_layer.h"
#include <exception>
#include <fstream>
#include <iomanip>
class NeuralNetwork
{
public:
	vector<NeuralLayer> layers;
	NeuralLayer* in_layer;
	NeuralLayer* out_layer;

	NeuralNetwork(vector<int> vi, unsigned int seed = 68441468)
	{
		layers.push_back(NeuralLayer(vi[0], nullptr));
		for (size_t i = 1; i < vi.size(); i++)
			layers.push_back(NeuralLayer(vi[i], &layers[i - 1]));
		in_layer = &layers[0];
		out_layer = &layers[layers.size() - 1];
		Neuron::random_w_engine.seed(seed);
	}
	NeuralNetwork(std::initializer_list<int> il) : NeuralNetwork(vector<int>(il)) {}
	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;

	void serialize(const char* filename)
	{
		std::ofstream sout;
		sout.open(filename);
		sout << std::setiosflags(std::ios::scientific) << std::setiosflags(std::ios::fixed);
		// first line : the number of layers
		sout << layers.size() << std::endl;
		// second line : the number of neurons
		for (size_t i = 0; i < layers.size(); i++)
			sout << layers[i].neurons.size() << " ";
		sout << std::endl;
		// next line : layer number, neuron number, activation, threshold, each weight
		for (size_t i = 1; i < layers.size(); i++)
			for (size_t j = 0; j < layers[i].neurons.size(); j++)
			{
				sout << i << "\t" << j << "\t";
				Neuron& n = layers[i].neurons[j];
				if (n.activation == n.sigmoid)
					sout << 1 << "\t";
				else if (n.activation == n.tanh)
					sout << 2 << "\t";
				else
					throw std::invalid_argument("unknown activation");
				sout << n.threshold << "\t";
				for (size_t k = 0; k < n.in.size(); k++)
					sout << n.w[k] << "\t";
				sout << std::endl;
			}
		sout.close();
	}

	// get info
	vector<int> size()
	{
		vector<int> vi;
		for (size_t i = 0; i < layers.size(); i++)
			vi.push_back(static_cast<int>(layers[i].neurons.size()));
	}
	int size_input() { return static_cast<int>(in_layer->neurons.size()); }
	int size_output() { return static_cast<int>(out_layer->neurons.size()); }

	// basic operate
	void set_in(vector<double> in)
	{
		if (in.size() != in_layer->neurons.size())
			throw std::invalid_argument("size(input number) != size(input neurons)");
		for (size_t i = 0; i < in.size(); i++)
			in_layer->neurons[i].out = in[i];
	}
	void update_out()
	{
		for (size_t i = 1; i < layers.size(); i++)
			for (size_t j = 0; j < layers[i].neurons.size(); j++)
				layers[i].neurons[j].update_out();
	}
	vector<double> get_out()
	{
		vector<double> out;
		for (size_t i = 0; i < out_layer->neurons.size(); i++)
			out.push_back(out_layer->neurons[i].out);
		return out;
	}
	void bp(double learning_rate, vector<double> correct_out)
	{
		if (correct_out.size() != out_layer->neurons.size())
			throw std::invalid_argument("size(output number) != size(output neurons)");
		for (size_t i = 0; i < correct_out.size(); i++)
		{
			double diff_error_out = out_layer->neurons[i].out - correct_out[i];
			out_layer->neurons[i].update_diff(diff_error_out);
			out_layer->neurons[i].bp(learning_rate);
		}
		for (size_t i = layers.size() - 2; i >= 1; i--)
			for (size_t j = 0; j < layers[i].neurons.size(); j++)
			{
				double diff_error_out = 0;
				for (size_t k = 0; k < layers[i + 1].neurons.size(); k++)
					diff_error_out += layers[i + 1].neurons[k].diff_error_in[j];
				layers[i].neurons[j].update_diff(diff_error_out);
				layers[i].neurons[j].bp(learning_rate);
			}
	}

	// user operate for one sample
	vector<double> test(vector<double> in)
	{
		set_in(in);
		update_out();
		return get_out();
	}
	double test_error(vector<double> in, vector<double> correct_out)
	{
		if (correct_out.size() != out_layer->neurons.size())
			throw std::invalid_argument("size(output number) != size(output neurons)");
		set_in(in);
		update_out();
		double error = 0;
		for (size_t i = 0; i < correct_out.size(); i++)
		{
			double offset = correct_out[i] - out_layer->neurons[i].out;
			error = offset * offset / 2;
		}
		return error;
	}
	void learning(double learning_rate, vector<double> in, vector<double> correct_out)
	{
		set_in(in);
		update_out();
		bp(learning_rate, correct_out);
	}
	void apply_learning(double rate=1.0)
	{
		for (size_t i = 1; i < layers.size(); i++)
			for (size_t j = 0; j < layers[i].neurons.size(); j++)
				layers[i].neurons[j].apply_new_w(rate);
	}

	// user operate for dataset
	double sum_error(vector<vector<double>> ins, vector<vector<double>> correct_outs)
	{
		if (ins.size() != correct_outs.size())
			throw std::invalid_argument("size(input of dateset) != size(output of dataset)");
		double error = 0;
		for (size_t i = 0; i < ins.size(); i++)
			error += test_error(ins[i], correct_outs[i]);
		return error;
	}
	void training(double learning_rate, vector<vector<double>> ins, vector<vector<double>> correct_outs)
	{
		if (ins.size() != correct_outs.size())
			throw std::invalid_argument("size(input of dateset) != size(output of dataset)");
		for (size_t i = 0; i < ins.size(); i++)
		{
			learning(learning_rate, ins[i], correct_outs[i]);
			apply_learning();
		}
	}
	double training_accumulate(double learning_rate, vector<vector<double>> ins, vector<vector<double>> correct_outs)
	{
		if (ins.size() != correct_outs.size())
			throw std::invalid_argument("size(input of dateset) != size(output of dataset)");
		double error = 0;
		for (size_t i = 0; i < ins.size(); i++)
		{
			error += test_error(ins[i], correct_outs[i]);
			bp(learning_rate, correct_outs[i]);
		}
		apply_learning(1.0 / ins.size());
		return error;
	}
};