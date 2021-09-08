#pragma once
#include "neural_layer.h"
#include <exception>
class NeuralNetwork
{
public:
	vector<NeuralLayer> layers;
	NeuralLayer* in_layer;
	NeuralLayer* out_layer;

	NeuralNetwork(vector<int> vi)
	{
		layers.push_back(NeuralLayer(vi[0], nullptr));
		for (size_t i = 1; i < vi.size(); i++)
			layers.push_back(NeuralLayer(vi[i], &layers[i - 1]));
		in_layer = &layers[0];
		out_layer = &layers[layers.size() - 1];
	}
	NeuralNetwork(std::initializer_list<int> il)
	{
		layers.push_back(NeuralLayer(*il.begin(), nullptr));
		for (auto it = il.begin() + 1; it != il.end(); it++)
			layers.push_back(NeuralLayer(*it, &layers[layers.size() - 1]));
		in_layer = &layers[0];
		out_layer = &layers[layers.size() - 1];
	}
	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;

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
		apply_learning();
		return error;
	}
};