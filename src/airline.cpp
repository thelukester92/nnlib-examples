#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

/*
nnlib:
0e0ba1855afba8bf44ed38758ec3896031d9acf9

nnlib-examples:
cfb75e85b2792d3290defaca417270cb22891c17

command:
./bin/airline -f ../../CaseStack/data/kh_kings-hawaiian-roll.arff -c 2 -e 1000 -s 400 -l 0.01 -d 1
*/

Tensor<> extrapolate(Sequencer<> &model, const Tensor<> &context, const Tensor<> &future)
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
	
	Tensor<> result(future.size(0), 1, 1);
	for(size_t i = 0; i < future.size(0); ++i)
	{
		Tensor<> inp(context.size(2));
		if(context.size(2) > 1)
		{
			inp.view(context.size(2) - 1).copy(future.narrow(0, i));
		}
		inp.narrow(0, context.size(2) - 1, 1).copy(model.output());
		
		result.narrow(0, i).copy(model.forward(inp.view(1, 1, inp.size())));
	}
	
	model.sequenceLength(sequenceLength);
	model.batch(bats);
	
	return result;
}

int main(int argc, const char **argv)
{
	ArgsParser args;
	args.addInt('s', "sequenceLength", 50);
	args.addInt('b', "batchSize", 20);
	args.addInt('e', "epochs", 100);
	args.addInt('n', "hiddenSize", 100);
	args.addInt('c', "seriesColumn", 0);
	args.addDouble('l', "learningRate", 0.01);
	args.addDouble('d', "learningRateDecay", 0.999);
	args.addDouble('v', "validationPart", 0.33);
	args.addString('f', "file", "data/airline.arff");
	args.parse(argc, argv);
	
	size_t sequenceLength		= std::max(args.getInt('s'), 1);
	size_t bats					= std::max(args.getInt('b'), 1);
	size_t epochs				= std::max(args.getInt('e'), 1);
	size_t hiddenSize			= std::max(args.getInt('n'), 1);
	size_t seriesColumn			= std::max(args.getInt('c'), 0);
	double validationPart		= std::min(std::max(args.getDouble('v'), 0.1), 0.9);
	double learningRate			= args.getDouble('l');
	double learningRateDecay	= args.getDouble('d');
	
	// Bootstrap
	
	cout << "===== Training on Airline =====" << endl;
	args.printOpts();
	cout << "Setting up..." << endl;
	
	RandomEngine::seed(0);
	cout << setprecision(5) << fixed;
	
	// Data loading
	
	Tensor<> series = File<>::loadArff(args.getString("file"));
	seriesColumn = std::min(seriesColumn, series.size(1));
	
	double min = series.narrow(1, seriesColumn).min(), max = series.narrow(1, seriesColumn).max();
	series.narrow(1, seriesColumn).normalize();
	
	// Data splitting
	
	size_t trainLength = (1.0 - validationPart) * series.size(0);
	size_t testLength = series.size(0) - trainLength + 1;
	
	Tensor<> train = series.narrow(0, 0, trainLength);
	Tensor<> test = series.narrow(0, trainLength - 1, testLength);
	
	Tensor<> future = test.narrow(1, 0, test.size(1) - 1).reshape(testLength, 1, test.size(1) - 1);
	
	Tensor<> trainFeat = train.narrow(0, 0, trainLength - 1);
	Tensor<> trainLab = train.narrow(0, 1, trainLength - 1).narrow(1, seriesColumn);
	
	Tensor<> testFeat = test.narrow(0, 0, testLength - 1);
	Tensor<> testLab = test.narrow(0, 1, testLength - 1).narrow(1, seriesColumn);
	
	// Modeling
	
	Sequencer<> nn(
		new Sequential<>(
			new LSTM<>(series.size(1), hiddenSize),
			new LSTM<>(50),
			new LSTM<>(20),
			new Linear<>(1)
		),
		sequenceLength
	);
	nn.batch(bats);
	
	CriticSequencer<> critic(new MSE<>(nn.module().outputs()), sequenceLength);
	Nadam<> optimizer(nn, critic);
	optimizer.learningRate(learningRate);
	
	Tensor<> preds = extrapolate(nn, trainFeat.view(trainLength - 1, 1, trainFeat.size(1)), future);
	double minError = critic.safeForward(preds, test.narrow(1, seriesColumn).view(testLength, 1, 1));
	cout << "Initial error: " << minError << endl;
	
	// Training
	
	SequenceBatcher<> batcher(trainFeat, trainLab, sequenceLength, bats);
	
	cout << "Training..." << endl;
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		
		nn.forget();
		optimizer.safeStep(batcher.features(), batcher.labels());
		optimizer.learningRate(optimizer.learningRate() * learningRateDecay);
		
		Progress<>::display(i, epochs);
		
		preds = extrapolate(nn, trainFeat.view(trainLength - 1, 1, trainFeat.size(1)), future);
		double err = critic.safeForward(preds, test.narrow(1, seriesColumn).view(testLength, 1, 1));
		
		Tensor<> seriesAndPreds(trainFeat.size(0) + preds.size(0), 3);
		seriesAndPreds.fill(File<>::unknown);
		seriesAndPreds.select(1, 0).narrow(0, 0, trainFeat.size(0)).copy(trainFeat.narrow(1, seriesColumn)).scale(max - min).shift(min);
		seriesAndPreds.select(1, 1).narrow(0, trainFeat.size(0), testLab.size(0)).copy(testLab).scale(max - min).shift(min);
		seriesAndPreds.select(1, 2).narrow(0, trainFeat.size(0), preds.size(0)).copy(preds).scale(max - min).shift(min);
		
		File<>::saveArff(seriesAndPreds, "pred-last.arff");
		if(err < minError)
		{
			minError = err;
			File<>::saveArff(seriesAndPreds, "pred-best.arff");
		}
		
		cout << "\terr: " << err << "\tmin: " << minError << flush;
	}
	Progress<>::display(epochs, epochs, '\n');
	
	preds = extrapolate(nn, trainFeat.view(trainLength - 1, 1, trainFeat.size(1)), future);
	critic.inputs(preds.shape());
	cout << "Final error: " << critic.safeForward(preds, test.narrow(1, seriesColumn).view(testLength, 1, 1)) << endl;
	
	return 0;
}
