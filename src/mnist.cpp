#include <iostream>
#include <nnlib/critics/nll.hpp>
#include <nnlib/nn/logsoftmax.hpp>
#include <nnlib/nn/sequential.hpp>
#include <nnlib/nn/linear.hpp>
#include <nnlib/nn/tanh.hpp>
#include <nnlib/opt/nadam.hpp>
#include <nnlib/serialization/csvserializer.hpp>
#include <nnlib/util/batcher.hpp>
#include <nnlib/util/progress.hpp>
using namespace nnlib;
using namespace std;

void load(const string &fname, Tensor<> &feat, Tensor<> &lab)
{
	Serialized rows = CSVSerializer::readFile(fname);
	
	feat.resize(rows.size(), rows.get(0)->size() - 1);
	lab.resize(rows.size(), 1);
	
	size_t i = 0;
	for(Serialized *row : rows.as<SerializedArray>())
	{
		for(size_t j = 0; j < row->size() - 1; ++j)
			feat(i, j) = row->get<double>(j) / 255.0;
		lab(i, 0) = row->get<double>(row->size() - 1);
		++i;
	}
}

size_t countMisclassifications(Module<> &nn, Tensor<> &feat, Tensor<> &lab)
{
	size_t miss = 0;
	for(size_t i = 0; i < feat.size(0); ++i)
	{
		nn.forward(feat.narrow(0, i));
		size_t pred = 0;
		for(size_t j = 1; j < nn.output().size(1); ++j)
			if(nn.output()(0, j) > nn.output()(0, pred))
				pred = j;
		if(pred != (size_t) lab(i, 0))
			++miss;
	}
	return miss;
}

int main()
{
	RandomEngine::seed(0);
	
	cout << "===== Training on MNIST =====" << endl;
	
	cout << "Loading data..." << flush;
	Tensor<> feat, lab, tFeat, tLab;
	load("data/mnist.train.csv", feat, lab);
	load("data/mnist.test.csv", tFeat, tLab);
	cout << " Done!" << endl;
	
	Sequential<> nn(
		new Linear<>(784, 300), new TanH<>(),
		new Linear<>(300, 100), new TanH<>(),
		new Linear<>(100, 10), new LogSoftMax<>()
	);
	NLL<> critic;
	Nadam<> optimizer(nn, critic);
	optimizer.learningRate(1e-3);
	
	Batcher<> batcher(feat, lab, 20);
	size_t batches = batcher.batches();
	size_t epochs = 5;
	
	for(size_t i = 0; i < epochs; ++i)
	{
		size_t j = 0;
		batcher.reset();
		do
		{
			optimizer.step(batcher.features(), batcher.labels());
			Progress::display(i * batches + j++ + 1, epochs * batches);
		}
		while(batcher.next());
	}
	
	cout << "Final misclassifications: " << countMisclassifications(nn, tFeat, tLab) << endl;
	
	return 0;
}
