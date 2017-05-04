#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

int main()
{
	RandomEngine::seed(0);
	
	cout << "===== Training on Airline =====" << endl;
	cout << "Setting up..." << endl;
	
	Tensor<double> series = File<>::loadArff("data/airline.arff");
	double min = series.min(), max = series.max();
	series.normalize();
	
	size_t trainLength = 0.67 * series.size(0);
	size_t testLength = series.size(0) - trainLength;
	
	Tensor<double> train = series.sub({ { 0, trainLength }, {} });
	Tensor<double> test = series.sub({ { trainLength, testLength }, {} });
	
	Tensor<double> trainFeat = train.sub({ { 0, trainLength - 1 }, {} });
	Tensor<double> trainLab = train.sub({ { 1, trainLength - 1 }, {} });
	
	Tensor<double> testFeat = test.sub({ { 0, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	Tensor<double> testLab = test.sub({ { 1, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	
	size_t seqLen = 10;
	size_t bats = 10;
	
	Sequencer<> nn(
		new Sequential<>(
			new Linear<>(1, 10), new TanH<>(),
			new Linear<>(10), new TanH<>(),
			new Linear<>(1)
		),
		seqLen
	);
	MSE<> critic(nn);
	auto optimizer = makeOptimizer<RMSProp>(nn, critic).learningRate(0.001);
	
	nn.seqLen(testFeat.size(0));
	nn.batch(1);
	critic.inputs(nn.outputs());
	cout << setprecision(5) << fixed;
	cout << "Initial error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	
	cout << "Training..." << endl;
	
	SequenceBatcher<> batcher(trainFeat, trainLab, seqLen, bats);
	nn.seqLen(batcher.seqLen());
	nn.batch(batcher.batch());
	critic.inputs(nn.outputs());
	
	size_t epochs = 1000;
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		Progress<>::display(i, epochs);
		optimizer.step(batcher.features(), batcher.labels());
	}
	Progress<>::display(epochs, epochs, '\n');
	
	nn.seqLen(testFeat.size(0));
	nn.batch(1);
	critic.inputs(nn.outputs());
	cout << setprecision(5) << fixed;
	cout << "Final error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	
	Tensor<> narrowed = testFeat.narrow(0, 0, 1);
	Tensor<> result = Tensor<>::flatten({ &train, &narrowed, &nn.forward(testFeat) });
	result.scale(max - min).shift(min);
	result.narrow(0, 0, train.size(0)).fill(File<>::unknown);
	result.resize(result.size(0), 1);
	
	File<>::saveArff(result, "pred.arff");
	
	return 0;
}
