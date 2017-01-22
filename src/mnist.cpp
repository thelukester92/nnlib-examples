#include <iostream>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

template <typename T = double>
Matrix<T> oneHot(const Matrix<T> &m, size_t outs)
{
	NNAssert(m.cols() == 1, "Can only one-hot a single-column matrix!");
	Matrix<T> out(m.rows(), outs, 0.0);
	for(size_t i = 0; i < m.rows(); ++i)
	{
		size_t closest = 0;
		for(size_t j = 1; j < outs; ++j)
			closest = fabs(m(i, 0) - j) > fabs(m(i, 0) - closest) ? j : closest;
		out(i, closest) = 1.0;
	}
	return out;
}

int main()
{
	Timer<> t;
	
	// Load the data
	
	cout << "Loading data... " << flush;
	t.reset();
	
	auto train	= Loader<>::loadArff("data/mnist_train.arff");
	auto test	= Loader<>::loadArff("data/mnist_test.arff");
	
	cout << "Done! Loaded " << train.rows() << " training samples and " << test.rows() << " testing samples in " << t.elapsed() << " seconds." << endl;
	
	// Preprocess the data
	
	cout << "Preprocessing data... " << flush;
	t.reset();
	
	auto trainFeat	= train.block(0, 0, train.rows(), train.cols() - 1);
	auto trainLab	= oneHot<>(train.block(0, train.cols() - 1, train.rows(), 1), 10);
	trainFeat.scale(1.0 / 255.0);
	
	auto testFeat	= test.block(0, 0, test.rows(), test.cols() - 1);
	auto testLab	= oneHot<>(test.block(0, test.cols() - 1, test.rows(), 1), 10);
	testFeat.scale(1.0 / 255.0);
	
	cout << "Done in " << t.elapsed() << " seconds." << endl;
	
	// Prepare the network and optimizer
	
	cout << "Preparing network and optimizer... " << flush;
	t.reset();
	
	Sequential<> nn;
	nn.add(
		new Linear<>(trainFeat.cols(), 300), new TanH<>(),
		new Linear<>(100), new TanH<>(),
		new Linear<>(10), new TanH<>()
	);
	
	SSE<> critic(10);
	auto optimizer = MakeOptimizer<RMSProp>(nn, critic);
	
	cout << "Done in " << t.elapsed() << " seconds." << endl;
	
	// Train
	
	cout << "Training... " << endl;
	
	nn.batch(testFeat.rows());
	critic.batch(testFeat.rows());
	cout << "Initial SSE: " << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
	
	size_t epochs = 50, batchesPerEpoch = 100;
	size_t batchSize = 25;
	Batcher<> batcher(trainFeat, trainLab, batchSize);
	nn.batch(batchSize);
	critic.batch(batchSize);
	
	t.reset();
	
	for(size_t i = 0; i < epochs; ++i)
	{
		for(size_t j = 0; j < batchesPerEpoch; ++j)
		{
			optimizer.optimize(batcher.features(), batcher.labels());
			batcher.next(true);
		}
		
		Progress::display(i, epochs);
	}
	Progress::display(epochs, epochs, '\n');
	
	double elapsed = t.elapsed();
	
	nn.batch(testFeat.rows());
	critic.batch(testFeat.rows());
	cout << "Final SSE: " << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
	
	cout << "Done in " << elapsed << " seconds." << endl;
	
	return 0;
}
