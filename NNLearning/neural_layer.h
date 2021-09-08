#pragma once
#include "neuron.h"
#include <utility>
class NeuralLayer
{
public:
	vector<Neuron> neurons;

	NeuralLayer(size_t size, NeuralLayer* prev_layer)
	{
		Neuron neuron;
		if (prev_layer != nullptr)
		{
			for (size_t i = 0; i < prev_layer->neurons.size(); i++)
				neuron.in.push_back(&prev_layer->neurons[i]);
			neuron.init_w();
		}
		neurons.insert(neurons.begin(), size, neuron);
	}
	NeuralLayer(const NeuralLayer&) = delete;
	NeuralLayer& operator=(const NeuralLayer&) = delete;
	NeuralLayer(NeuralLayer&& layer) noexcept : neurons(std::move(layer.neurons)) {}
	NeuralLayer& operator=(NeuralLayer&&) = delete;
};