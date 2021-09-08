#include <iostream>
#include <fstream>
#include "neural_network.h"

using std::cin;
using std::cout;
using std::endl;

int main()
{
	std::ofstream fout;
	fout.open("out.txt");

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
	NeuralNetwork nn{ 1,8,8,1 };
	cout << "learning..." << endl;
	for (int i = 0; i < 10000; i++) // loop
	{
		double error = 0;
		for (double in = 0; in < 10; in += 0.1) // 100 points
		{
			double out = std::sin(in);
			error += nn.test_error(vector<double>{in}, vector<double>{out});
		}
		cout << "current sum error : " << error << endl;
		for (double in = 0; in < 10; in += 0.1) // 100 points
		{
			double out = std::sin(in);
			nn.learning(0.03, vector<double>{in}, vector<double>{out});
			nn.apply_learning();
		}
	}
	cout << "testing..." << endl;
	fout << "test in  \t| out(correct out)" << endl;
	for (double in = 0; in < 10; in += 0.05) // 200 points
	{
		double correct_out = std::sin(in);
		vector<double> out = nn.test(vector<double>{in});
		fout << "test " << in << "  \t| " << out[0] << "(" << correct_out << ")" << endl;
	}

	fout.close();
	return 1;
}