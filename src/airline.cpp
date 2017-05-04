#include <iostream>
#include <iomanip>
#include <nnlib.h>
using namespace std;
using namespace nnlib;

int main()
{
	LSTM<> nn(1, 1);
	Tensor<> params = Tensor<>::flatten(nn.parameters());
	
	params.copy({
		// input gate
		0.2,	0,
		-0.1,	0,
		0.25,	0,
		
		// forget gate
		-0.2,	0,
		0.5,	0,
		1.0,	0,
		
		// input module
		0.1,	0,
		0.2,	0,
		
		// output gate
		0.5,	0,
		0.1,	0,
		0.3,	0
	});
	
	Tensor<> sequence = { 8, 6, 7, 5, 3, 0, 9, 1, 2, 4 };
	sequence.resize(sequence.size(0), 1, 1);
	
	Tensor<> state = Tensor<>::flatten(nn.innerState());
	Tensor<> states(sequence.size(0), state.size(0));
	
	for(size_t i = 0; i < sequence.size(0); ++i)
	{
		nn.forward(sequence.select(0, i));
		states.select(0, i).copy(state);
		
		/*
		cout << nn.m_inpGate->output() << endl;
		cout << nn.m_fgtGate->output() << endl;
		cout << nn.m_inpMod->output() << endl;
		cout << nn.m_state << endl;
		cout << nn.m_outGate->output() << endl;
		cout << nn.output() << endl;
		cout << endl;
		*/
		
		if(i == 2)
			break;
	}
	
	for(size_t i = 3; i > 0; --i)
	{
		state.copy(states.select(0, i - 1));
		
		Tensor<> grad = Tensor<>({ nn.output()(0, 0) - sequence(i, 0, 0) }).resize(1, 1);
		nn.backward(sequence.select(0, i - 1), grad);
		
		cout << nn.output() << endl;
		cout << grad << endl;
		cout << nn.m_outMod->inGrad() << endl;
		cout << nn.m_outGate->inGrad() << endl;
		cout << nn.m_curStateGrad << endl;
		cout << nn.m_inpMod->inGrad() << endl;
		cout << nn.m_inpGate->inGrad() << endl;
		cout << nn.m_fgtGate->inGrad() << endl;
		cout << nn.inGrad() << endl;
		cout << nn.m_outGrad << endl;
		cout << nn.m_stateGrad << endl;
		cout << endl;
		
		if(i-1 == 1)
			break;
	}
	
	/*
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
	
	train = series;
	trainLength = train.size(0);
	
	test = series;
	testLength = test.size(0);
	
	Tensor<double> trainFeat = train.sub({ { 0, trainLength - 1 }, {} });
	Tensor<double> trainLab = train.sub({ { 1, trainLength - 1 }, {} });
	
	Tensor<double> testFeat = test.sub({ { 0, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	Tensor<double> testLab = test.sub({ { 1, testLength - 1 }, {} }).resize(testLength - 1, 1, 1);
	
	size_t seqLen = 10;
	size_t bats = 10;
	
	Sequencer<> nn(
		new Sequential<>(
			new Linear<>(1, 10), new TanH<>(),
			new LSTM<>(10),
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
	
	size_t epochs = 10000;
	for(size_t i = 0; i < epochs; ++i)
	{
		nn.seqLen(trainFeat.size(0));
		nn.batch(1);
		nn.forget();
		//nn.forward(trainFeat);
		nn.seqLen(testFeat.size(0));
		critic.inputs(nn.outputs());
		cout << setprecision(5) << fixed;
		cout << "Final error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
		nn.seqLen(batcher.seqLen());
		nn.batch(batcher.batch());
		critic.inputs(nn.outputs());
		
		batcher.reset();
		// Progress<>::display(i, epochs);
		nn.forget();
		optimizer.step(batcher.features(), batcher.labels());
	}
	Progress<>::display(epochs, epochs, '\n');
	
	nn.seqLen(trainFeat.size(0));
	nn.batch(1);
	nn.forget();
	//nn.forward(trainFeat);
	nn.seqLen(testFeat.size(0));
	critic.inputs(nn.outputs());
	cout << setprecision(5) << fixed;
	cout << "Final error: " << critic.forward(nn.forward(testFeat), testLab) << endl;
	
	Tensor<> result = Tensor<>::flatten({ &nn.forward(testFeat) });
	result.scale(max - min).shift(min);
	result.resize(result.size(0), 1);
	
	File<>::saveArff(result, "pred.arff");
	*/
	
	return 0;
}
