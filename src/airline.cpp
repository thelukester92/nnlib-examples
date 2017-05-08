#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

Tensor<> extrapolate(Sequencer<> &model, const Tensor<> &context, size_t length)
{
	model.forget();
	model.seqLen(1);
	
	for(size_t i = 0; i < context.size(0); ++i)
	{
		model.forward(context.narrow(0, i));
	}
	
	Tensor<> result(length, 1, 1);
	for(size_t i = 0; i < length; ++i)
	{
		result.narrow(0, i).copy(model.forward(model.output()));
	}
	
	return result.resize(result.size(0), 1);
}

int main()
{
	RandomEngine::seed(0);
	
	cout << "===== Training on Airline =====" << endl;
	cout << "Setting up..." << endl;
	
	Tensor<double> series = File<>::loadArff("data/airline.arff");
	double min = series.min(), max = series.max();
	series.normalize();
	
	size_t trainLength = 0.67 * series.size(0);
	size_t testLength = series.size(0) - trainLength + 1;
	
	Tensor<double> train = series.sub({ { 0, trainLength }, {} });
	Tensor<double> test = series.sub({ { trainLength - 1, testLength }, {} });
	
	Tensor<double> trainFeat = train.sub({ { 0, trainLength - 1 }, {} });
	Tensor<double> trainLab = train.sub({ { 1, trainLength - 1 }, {} });
	
	Tensor<double> testFeat = test.sub({ { 0, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	Tensor<double> testLab = test.sub({ { 1, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	
	size_t seqLen = 10;
	size_t bats = 10;
	
	Sequencer<> nn(
		new Sequential<>(
			new LSTM<>(1, 50),
			new LSTM<>(50),
			new Linear<>(1)
		),
		seqLen
	);
	MSE<> critic(nn);
	auto optimizer = makeOptimizer<SGD>(nn, critic).learningRate(0.1);
	
	nn.seqLen(testFeat.size(0));
	nn.batch(1);
	critic.inputs(nn.outputs());
	cout << setprecision(5) << fixed;
	cout << "Initial error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	
	cout << "Training..." << endl;
	
	nn.seqLen(series.size(0) - 1);
	nn.batch(1);
	critic.inputs(nn.outputs());
	
	Tensor<> fromSeries = series.sub({ { 0, series.size(0) - 1 }, {} }).resize(series.size(0) - 1, 1, 1);
	Tensor<> toSeries = series.sub({ { 1, series.size(0) - 1 }, {} }).resize(series.size(0) - 1, 1, 1);
	
	size_t epochs = 200;
	for(size_t i = 0; i < epochs; ++i)
	{
		nn.forget();
		optimizer.step(fromSeries, toSeries);
		optimizer.learningRate(optimizer.learningRate() * 0.999);
		Progress<>::display(i, epochs);
	}
	Progress<>::display(epochs, epochs, '\n');
	cout << setprecision(5) << critic.forward(nn.forward(fromSeries), toSeries) << endl;
	
	Tensor<> preds = extrapolate(nn, series.narrow(0, 0, trainLength), testLength);
	Tensor<> seriesAndPreds = Tensor<>::flatten({ &trainFeat, &preds })
		.resize(series.size(0), 1)
		.scale(max - min)
		.shift(min);
	
	File<>::saveArff(seriesAndPreds, "pred.arff");
	
	return 0;
}
