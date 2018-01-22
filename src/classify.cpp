#include <iostream>
#include <nnlib/critics/nll.hpp>
#include <nnlib/math/math.hpp>
#include <nnlib/nn/logsoftmax.hpp>
#include <nnlib/nn/sequential.hpp>
#include <nnlib/nn/linear.hpp>
#include <nnlib/nn/tanh.hpp>
#include <nnlib/opt/nadam.hpp>
#include <nnlib/serialization/fileserializer.hpp>
#include <nnlib/util/args.hpp>
#include <nnlib/util/batcher.hpp>
#include <nnlib/util/progress.hpp>
#include <unordered_set>
using namespace nnlib;
using namespace std;
using T = double;

void load(const string &fname, Tensor<T> &feat, Tensor<T> &lab)
{
    feat = Tensor<T>(FileSerializer::read(fname));
    lab  = feat.narrow(1, feat.size(1) - 1);
    feat = feat.narrow(1, 0, feat.size(1) - 1) / 255.0;
}

int main(int argc, const char **argv)
{
    ArgsParser args;
    args.addInt('b', "batch", 20);
    args.addInt('e', "epochs", 5);
    args.addDouble('l', "learningRate", 1e-3);
    args.addInt('s', "seed", -1);
    args.addString("train", "data/mnist.train.bin");
    args.addString("test", "data/mnist.test.bin");
    args.parse(argc, argv);
    args.printOpts();

    size_t batch   = args.getInt("batch");
    size_t epochs  = args.getInt("epochs");
    double lr      = args.getDouble("learningRate");
    int seed       = args.getInt("seed");
    string train   = args.getString("train");
    string test    = args.getString("test");

    if(seed >= 0)
        RandomEngine::sharedEngine().seed(seed);

    clog << "Loading data..." << flush;
    Tensor<T> feat, lab, tFeat, tLab;
    load(train, feat, lab);
    load(test, tFeat, tLab);
    size_t inps = feat.size(1);
    size_t outs = math::max(lab);
    clog << " Done! Mapping " << inps << " -> " << outs << "!" << endl;

    Sequential<T> nn(
        new Linear<T>(inps, 300), new TanH<T>(),
        new Linear<T>(300, 100), new TanH<T>(),
        new Linear<T>(100, outs), new LogSoftMax<T>()
    );

    NLL<T> *critic = new NLL<T>();
    Nadam<T> optimizer(nn, critic);
    optimizer.learningRate(lr);

    Batcher<T> batcher(feat, lab, batch);
    size_t batches = batcher.batches();

    Progress p(epochs * batches, clog);
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
