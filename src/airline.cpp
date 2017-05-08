#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

Tensor<> extrapolate(Sequencer<> &model, const Tensor<> &context, size_t length)
{
	size_t sequenceLength = model.sequenceLength();
	size_t bats = model.batch();
	
	model.forget();
	model.sequenceLength(1);
	model.batch(1);
	
	for(size_t i = 0; i < context.size(0); ++i)
	{
		model.forward(context.narrow(0, i));
	}
	
	Tensor<> result(length, 1, 1);
	for(size_t i = 0; i < length; ++i)
	{
		result.narrow(0, i).copy(model.forward(model.output()));
	}
	
	model.sequenceLength(sequenceLength);
	model.batch(bats);
	
	return result;
}

int main()
{
	// Metaparameters
	
	size_t sequenceLength = 50;
	size_t bats = 10;
	size_t epochs = 1000;
	double validationPart = 0.33;
	double learningRate = 0.01 / bats;
	
	// Bootstrap
	
	RandomEngine::seed(0);
	cout << setprecision(5) << fixed;
	
	cout << "===== Training on Airline =====" << endl;
	cout << "Setting up..." << endl;
	
	// Data loading
	
	Tensor<> series = File<>::loadArff("data/airline.arff");
	double min = series.min(), max = series.max();
	series.normalize();
	
	// Data splitting
	
	size_t trainLength = (1.0 - validationPart) * series.size(0);
	size_t testLength = series.size(0) - trainLength + 1;
	
	Tensor<> train = series.narrow(0, 0, trainLength);
	Tensor<> test = series.narrow(0, trainLength - 1, testLength);
	
	Tensor<> trainFeat = train.narrow(0, 0, trainLength - 1);
	Tensor<> trainLab = train.narrow(0, 1, trainLength - 1);
	
	Tensor<> testFeat = test.narrow(0, 0, testLength - 1);
	Tensor<> testLab = test.narrow(0, 1, testLength - 1);
	
	// Modeling
	
	Sequencer<> nn(
		new Sequential<>(
			new LSTM<>(1, 20),
			new Linear<>(1)
		),
		sequenceLength
	);
	nn.batch(bats);
	
	MSE<> critic(nn.outputs());
	SGD<> optimizer(nn, critic);
	optimizer.learningRate(learningRate);
	
	Tensor<> preds = extrapolate(nn, trainFeat.reshape(trainLength - 1, 1, 1), testLength);
	cout << "Initial error: " << critic.forward(preds, test.reshape(testLength, 1, 1)) << endl;
	
	// Training
	
	SequenceBatcher<> batcher(trainFeat, trainLab, sequenceLength, bats);
	
	// nn.batch(1);
	// critic.inputs(nn.outputs());
	
	cout << "Training..." << endl;
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		
		// optimizer.step(batcher.features(), batcher.labels());
		
		/*
		for(size_t k = 0; k < bats; ++k)
		{
			optimizer.step(batcher.features().narrow(1, k), batcher.features().narrow(1, k));
		}
		*/
		
		Tensor<> inps(nn.inputs(), true);
		Tensor<> outs(nn.outputs(), true);
		Tensor<> grads(optimizer.grads().shape(), true);
		grads.fill(0);
		
		nn.batch(1);
		critic.inputs(nn.outputs());
		
		for(size_t k = 0; k < bats; ++k)
		{
			optimizer.grads().fill(0);
			nn.forward(batcher.features().narrow(1, k));
			critic.backward(nn.output(), batcher.labels().narrow(1, k));
			nn.backward(batcher.features().narrow(1, k), critic.inGrad());
			grads.addVV(optimizer.grads());
		}
		
		nn.batch(bats);
		critic.inputs(nn.outputs());
		
		optimizer.grads().fill(0);
		nn.forward(batcher.features());
		critic.backward(nn.output(), batcher.labels());
		nn.backward(batcher.features(), critic.inGrad());
		
		cout << setprecision(10);
		cout << MSE<>(grads.shape()).forward(grads, optimizer.grads()) * grads.size() << endl;
		return 0;
		
		/*
		nn.forget();
		optimizer.step(batcher.features(), batcher.labels());
		optimizer.learningRate(optimizer.learningRate() * (1.0 - 1e-6));
		*/
		
		Progress<>::display(i, epochs);
	}
	Progress<>::display(epochs, epochs, '\n');
	
	preds = extrapolate(nn, trainFeat.reshape(trainLength - 1, 1, 1), testLength);
	
	cout << "Final error: " << critic.forward(preds, test.reshape(testLength, 1, 1)) << endl;
	
	Tensor<> seriesAndPreds(trainFeat.size(0) + preds.size(0), 3);
	seriesAndPreds.fill(File<>::unknown);
	seriesAndPreds.select(1, 0).narrow(0, 0, trainFeat.size(0)).copy(trainFeat).scale(max - min).shift(min);
	seriesAndPreds.select(1, 1).narrow(0, trainFeat.size(0), testLab.size(0)).copy(testLab).scale(max - min).shift(min);
	seriesAndPreds.select(1, 2).narrow(0, trainFeat.size(0), preds.size(0)).copy(preds).scale(max - min).shift(min);
	
	File<>::saveArff(seriesAndPreds, "pred.arff");
	
	return 0;
}
