#include <nnlib.h>
#include <iostream>
#include <chrono>
using namespace nnlib;
using namespace std;

int main(int argc, const char **argv)
{
	using clock = chrono::high_resolution_clock;
	
	ArgsParser args;
	args.addInt('d', "depth", 10);
	args.addInt('w', "width", 1000);
	args.addInt('i', "inputs", 100);
	args.addInt('o', "outputs", 10);
	args.addInt('b', "batch", 100);
	args.addInt('s', "steps", 100);
	args.parse(argc, argv);
	args.printOpts();
	
	size_t depth = args.getInt('d');
	size_t width = args.getInt('w');
	size_t inps = args.getInt('i');
	size_t outs = args.getInt('o');
	size_t bats = args.getInt('b');
	size_t steps = args.getInt('s');
	
	NNAssert(depth > 2, "Depth must be greater than 2!");
	
	Sequential<> nn(new Linear<>(width), new TanH<>());
	for(size_t i = 0; i < depth - 2; ++i)
		nn.add(new Linear<>(width), new TanH<>());
	nn.add(new Linear<>(outs));
	nn.batch(bats);
	
	Tensor<> input = Tensor<>(bats, inps);
	Tensor<> grad = Tensor<>(bats, outs);
	
	auto start = clock::now();
	for(size_t i = 0; i < steps; ++i)
	{
		input.rand();
		grad.rand();
		
		nn.safeForward(input);
		nn.safeBackward(input, grad);
	}
	cout << "Finished " << steps << " steps in " << chrono::duration<double>(clock::now() - start).count() << endl;
	
	return 0;
}
