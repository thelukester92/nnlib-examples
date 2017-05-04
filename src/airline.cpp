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
	
	Tensor<double> testFeat = test.sub({ { 0, testLength - 1 }, {} });
	Tensor<double> testLab = test.sub({ { 1, testLength - 1 }, {} });
	
	Sequential<> nn(
		new Linear<>(1, 10), new TanH<>(),
		new Linear<>(10), new TanH<>(),
		new Linear<>(1)
	);
	MSE<> critic(nn);
	auto optimizer = makeOptimizer<RMSProp>(nn, critic).learningRate(0.001);
	
	nn.batch(testFeat.size(0));
	critic.batch(testLab.size(0));
	cout << "Initial error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	cout << "Training..." << endl;
	
	Batcher<> batcher(trainFeat, trainLab, 10, true);
	nn.batch(batcher.batch());
	critic.batch(batcher.batch());
	
	size_t epochs = 100;
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
	
	Tensor<> fullResult(series.size(), 1);
	fullResult.concat(
		0,
		train,
		testFeat.narrow(0, 0, 1),
		nn.forward(testFeat)
	);
	fullResult.scale(max - min).shift(min);
	fullResult.sub({ { 0, train.size(0) }, {} }).fill(File<>::unknown);
	
	File<>::saveArff(fullResult, "pred.arff");
	
	return 0;
}
