#include <iostream>
#include <fstream>
#include <thread>
#include "neural_network.h"

using std::cin;
using std::cout;
using std::endl;

int main()
{
	cout << "build dateset and testset..." << endl;
	vector<vector<double>> data_ins, data_outs, test_ins, test_outs;
	for (double in = 0; in < 10; in += 0.1) // 100 points
	{
		double out = std::sin(in);
		data_ins.push_back(vector<double>{in});
		data_outs.push_back(vector<double>{out});
	}
	for (double in = 0.05; in < 10; in += 0.1) // 100 points
	{
		double out = std::sin(in);
		test_ins.push_back(vector<double>{in});
		test_outs.push_back(vector<double>{out});
	}

	cout << "NN..." << endl;
	NeuralNetwork nn(NeuralNetwork::deserialize("nn.txt"));
	//NeuralNetwork nn{ 1,8,8,1 };

	cout << "press any key to start learning or stop learning" << endl;
	cin.ignore();
	bool stop_training_func = false;
	int training_times;
	auto training_func = [&stop_training_func,&training_times, &nn, &data_ins, &data_outs, &test_ins, &test_outs]()
	{
		training_times = 0;
		while (!stop_training_func) // loop
		{
			cout << "current data error : " << nn.sum_error(data_ins, data_outs) << "\t";
			cout << "current test error : " << nn.sum_error(test_ins, test_outs) << endl;
			nn.training_accumulate(0.03, data_ins, data_outs);
			training_times++;
		}
	};
	cout << "learning..." << endl;
	std::thread training_thread(training_func);
	cin.ignore();
	stop_training_func = true;
	training_thread.join();
	cout << "training " << training_times << " times" << endl;

	cout << "serialize NN and test NN?(y/n)" << endl;
	while (true)
	{
		char c;
		cin >> c;
		if (c == 'y' || c == 'Y')
			break;
		else if (c == 'n' || c == 'N')
			return 1;
	}

	nn.serialize("nn.txt");

	cout << "testing..." << endl;
	std::ofstream fout;
	fout.open("out.txt");
	fout << "test in  \t| out(correct out)" << endl;
	for (double in = 0; in < 10; in += 0.05) // 200 points
	{
		double correct_out = std::sin(in);
		vector<double> out = nn.test(vector<double>{in});
		fout << "test " << in << "  \t| " << out[0] << "(" << correct_out << ")" << endl;
	}
	fout.close();

	return 0;
}