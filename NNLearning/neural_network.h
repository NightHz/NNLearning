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
	void bp(double learning_velocity, vector<double> correct_out)
	{
		if (correct_out.size() != out_layer->neurons.size())
			throw std::invalid_argument("size(output number) != size(output neurons)");
		for (size_t i = 0; i < correct_out.size(); i++)
		{
			double diff_error_out = out_layer->neurons[i].out - correct_out[i];
			out_layer->neurons[i].update_diff(diff_error_out);
			out_layer->neurons[i].bp(learning_velocity);
		}
		for (size_t i = layers.size() - 2; i >= 1; i--)
			for (size_t j = 0; j < layers[i].neurons.size(); j++)
			{
				double diff_error_out = 0;
				for (size_t k = 0; k < layers[i + 1].neurons.size(); k++)
					diff_error_out += layers[i + 1].neurons[k].diff_error_in[j];
				layers[i].neurons[j].update_diff(diff_error_out);
				layers[i].neurons[j].bp(learning_velocity);
			}
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
			error = offset * offset;
		}
		return error;
	}
	void learning(double learning_velocity, vector<double> in, vector<double> correct_out)
	{
		set_in(in);
		update_out();
		bp(learning_velocity, correct_out);
	}
	void apply_learning();
	void training(double learning_velocity, vector<vector<double>> ins, vector<vector<double>> correct_outs, bool accumulate = false)
	{
		if (ins.size() != correct_outs.size())
			throw std::invalid_argument("size(input of dateset) != size(output of dataset)");
		for (size_t i = 0; i < ins.size(); i++)
		{
			learning(learning_velocity, ins[i], correct_outs[i]);
			if (!accumulate)
				apply_learning();
		}
		if (accumulate)
			apply_learning();
	}
};