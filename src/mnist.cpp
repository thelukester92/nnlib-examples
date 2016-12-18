#include <iostream>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

int main()
{
	// Load the data
	
	cout << "Loading data... " << flush;
	
	auto train	= Loader<>::loadArff("data/mnist_train.arff");
	auto test	= Loader<>::loadArff("data/mnist_test.arff");
	
	cout << "Done!" << endl;
	
	// Preprocess the data
	
	cout << "Preprocessing data... " << flush;
	
	auto trainFeat	= train.block(0, 0, train.rows(), train.cols() - 1);
	auto trainLab	= Matrix<>(train.rows(), 10, 0);
	trainFeat.scale(1.0 / 255.0);
	
	auto testFeat	= test.block(0, 0, test.rows(), test.cols() - 1);
	auto testLab	= Matrix<>(test.rows(), 10, 0);
	testFeat.scale(1.0 / 255.0);
	
	for(size_t i = 0; i < trainLab.rows(); ++i)
		trainLab[train[i].back()] = 1;
	
	for(size_t i = 0; i < testLab.rows(); ++i)
		testLab[test[i].back()] = 1;
	
	cout << "Done!" << endl;
	
	// Prepare the network and optimizer
	
	cout << "Preparing network and optimizer... " << flush;
	
	Sequential<> nn;
	nn.add(
		new Linear<>(trainFeat.cols(), 300), new TanH<>(),
		new Linear<>(100), new TanH<>(),
		new Linear<>(10), new TanH<>()
	);
	
	SSE<> critic(10);
	auto optimizer = MakeOptimizer<RMSProp>(nn, critic);
	
	cout << "Done!" << endl;
	
	// Train
	
	cout << "Training... " << endl;
	
	nn.batch(testFeat.rows());
	critic.batch(testFeat.rows());
	cout << "Initial SSE: " << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
	
	size_t epochs = 50, batchesPerEpoch = 100;
	size_t batchSize = 10;
	Batcher<> batcher(trainFeat, trainLab, batchSize);
	nn.batch(batchSize);
	critic.batch(batchSize);
	
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
	
	nn.batch(testFeat.rows());
	critic.batch(testFeat.rows());
	cout << "Final SSE: " << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
	
	cout << "Done!" << endl;
	
	return 0;
}
