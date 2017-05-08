#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

size_t countMisclassifications(Module<> &model, const Tensor<> &feat, const Tensor<> &lab)
{
	model.batch(feat.size(0));
	model.forward(feat);
	
	size_t misclassifications = 0;
	for(size_t i = 0; i < feat.size(0); ++i)
	{
		size_t max = 0;
		for(size_t j = 1; j < model.output().size(1); ++j)
		{
			if(model.output()(i, j) > model.output()(i, max))
				max = j;
		}
		if(max != (size_t) lab(i, 0))
			++misclassifications;
	}
	
	return misclassifications;
}

int main()
{
	RandomEngine::seed(0);
	
	cout << "===== Training on MNIST =====" << endl;
	cout << "Setting up..." << endl;
	
	Relation rel;
	Tensor<double> train = File<>::loadArff("data/mnist.train.arff", &rel);
	Tensor<double> test = File<>::loadArff("data/mnist.test.arff");
	size_t outs = rel.attrVals(rel.size() - 1).size();
	
	Tensor<double> trainFeat = train.sub({ {}, { 0, train.size(1) - 1 } }).scale(1.0 / 255.0);
	Tensor<double> trainLab = train.sub({ {}, { train.size(1) - 1 } });
	
	Tensor<double> testFeat = test.sub({ {}, { 0, test.size(1) - 1 } }).scale(1.0 / 255.0);
	Tensor<double> testLab = test.sub({ {}, { test.size(1) - 1 } });
	
	Sequential<> nn(
		new Linear<>(trainFeat.size(1), 300), new TanH<>(),
		new Linear<>(100), new TanH<>(),
		new Linear<>(outs), new TanH<>(),
		new LogSoftMax<>()
	);
	NLL<> critic(nn.outputs());
	RMSProp<> optimizer(nn, critic);
	optimizer.learningRate(0.001);
	
	nn.batch(testFeat.size(0));
	critic.batch(testLab.size(0));
	cout << "Initial error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	cout << "Initial miss:  " << countMisclassifications(nn, testFeat, testLab) << endl;
	cout << "Training..." << endl;
	
	Batcher<> batcher(trainFeat, trainLab, 100);
	nn.batch(batcher.batch());
	critic.batch(batcher.batch());
	
	size_t epochs = 10;
	size_t k = 0, tot = epochs * batcher.batches();
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		do
		{
			Progress<>::display(k++, tot);
			optimizer.step(batcher.features(), batcher.labels());
		}
		while(batcher.next());
	}
	Progress<>::display(tot, tot, '\n');
	
	nn.batch(testFeat.size(0));
	critic.batch(testLab.size(0));
	cout << setprecision(5) << fixed;
	cout << "Final error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	cout << "Final miss:  " << countMisclassifications(nn, testFeat, testLab) << endl;
	
	return 0;
}
