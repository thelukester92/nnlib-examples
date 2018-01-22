#include <iostream>
#include <nnlib/critics/nll.hpp>
#include <nnlib/math/math.hpp>
#include <nnlib/nn/logsoftmax.hpp>
#include <nnlib/nn/sequential.hpp>
#include <nnlib/nn/linear.hpp>
#include <nnlib/nn/tanh.hpp>
#include <nnlib/opt/nadam.hpp>
#include <nnlib/serialization/fileserializer.hpp>
#include <nnlib/util/batcher.hpp>
#include <nnlib/util/progress.hpp>
using namespace nnlib;
using namespace std;
using T = double;

void load(const string &fname, Tensor<T> &feat, Tensor<T> &lab)
{
    feat = Tensor<T>(FileSerializer::read(fname));
    lab  = feat.narrow(1, feat.size(1) - 1);
    feat = feat.narrow(1, 0, feat.size(1) - 1) / 255.0;
}

int main()
{
    RandomEngine::sharedEngine().seed(0);

    cout << "===== Training on MNIST =====" << endl;

    cout << "Loading data..." << flush;
    Tensor<T> feat, lab, tFeat, tLab;
    load("data/mnist.train.bin", feat, lab);
    load("data/mnist.test.bin", tFeat, tLab);
    cout << " Done!" << endl;

    Sequential<T> nn(
        new Linear<T>(784, 300), new TanH<T>(),
        new Linear<T>(300, 100), new TanH<T>(),
        new Linear<T>(100, 10), new LogSoftMax<T>()
    );

    NLL<T> *critic = new NLL<T>();
    Nadam<T> optimizer(nn, critic);
    optimizer.learningRate(1e-3);

    Batcher<T> batcher(feat, lab, 20);
    size_t batches = batcher.batches();
    size_t epochs = 5;

    Progress p(epochs * batches);
    p.display(0);

    for(size_t i = 0; i < epochs; ++i)
    {
        size_t j = 0;
        batcher.reset();
        do
        {
            optimizer.step(batcher.features(), batcher.labels());
            p.display(i * batches + ++j);
        }
        while(batcher.next());
    }

    cout << "Final misclassifications: " << critic->misclassifications(nn.forward(tFeat), tLab) << endl;

    return 0;
}
