#include <iostream>
#include <iomanip>
#include <nnlib/core/tensor.hpp>
#include <nnlib/critics/criticsequencer.hpp>
#include <nnlib/critics/mse.hpp>
#include <nnlib/math/math.hpp>
#include <nnlib/math/random.hpp>
#include <nnlib/nn/linear.hpp>
#include <nnlib/nn/sequencer.hpp>
#include <nnlib/nn/sequential.hpp>
#include <nnlib/nn/lstm.hpp>
#include <nnlib/opt/nadam.hpp>
#include <nnlib/serialization/fileserializer.hpp>
#include <nnlib/util/args.hpp>
#include <nnlib/util/batcher.hpp>
#include <nnlib/util/progress.hpp>
using namespace std;
using namespace nnlib;
using namespace nnlib::math;
using T = double;

void load(const std::string &infile, double validationPart, Tensor<T> &feat, Tensor<T> &lab, Tensor<T> &tFeat, Tensor<T> &tLab, T &min, T &max)
{
    Tensor<T> series = Tensor<T>(FileSerializer::read(infile));
    min = math::min(series.narrow(1, 0));
    max = math::max(series.narrow(1, 0));
    normalize(series.narrow(1, 0));

    size_t trainLength = (1.0 - validationPart) * series.size(0);
    size_t testLength = series.size(0) - trainLength + 1;

    Tensor<T> train = series.narrow(0, 0, trainLength);
    Tensor<T> test = series.narrow(0, trainLength - 1, testLength);

    feat = train.narrow(0, 0, trainLength - 1);
    lab = train.narrow(0, 1, trainLength - 1).narrow(1, 0);
    tFeat = test.narrow(0, 0, testLength - 1);
    tLab = test.narrow(0, 1, testLength - 1).narrow(1, 0);
}

void extrapolate(Sequencer<T> &model, const Tensor<T> &context, Tensor<T> &preds)
{
    model.forget();
    for(size_t i = 0; i < context.size(0); ++i)
        model.forward(context.narrow(0, i));
    for(size_t i = 0; i < preds.size(0); ++i)
        preds(0, 0, 0) = model.forward(model.output())(0, 0, 0);
}

int main(int argc, const char **argv)
{
    ArgsParser args;
    args.addInt('b', "batchSize", 20);
    args.addInt('e', "epochs", 100);
    args.addString('i', "infile", "data/airline.bin");
    args.addDouble('l', "learningRate", 0.01);
    args.addInt('n', "hiddenSize", 100);
    args.addString('o', "predictionOut", "pred-last.arff");
    args.addInt('s', "sequenceLength", 50);
    args.addDouble('v', "validationPart", 0.33);
    args.addInt("seed", -1);
    args.parse(argc, argv);
    args.printOpts();

    size_t bats           = args.getInt('b');
    size_t epochs         = args.getInt('e');
    size_t hiddenSize     = args.getInt('n');
    size_t sequenceLength = args.getInt('s');
    T learningRate        = args.getDouble('l');
    T validationPart      = args.getDouble('v');
    string infile         = args.getString("infile");
    int seed              = args.getInt("seed");

    if(seed >= 0)
        RandomEngine::sharedEngine().seed(seed);

    Tensor<T> feat, lab, tFeat, tLab;
    T min, max;
    load(infile, validationPart, feat, lab, tFeat, tLab, min, max);

    Tensor<T> feat3D = feat.view(feat.size(0), 1, feat.size(1));
    Tensor<T> tLab3D = tLab.view(tLab.size(0), 1, tLab.size(1));

    Sequencer<T> nn(
        new Sequential<T>(
            &(new LSTM<T>(feat.size(1), hiddenSize))->gradClip(10),
            &(new LSTM<T>(hiddenSize, 50))->gradClip(10),
            &(new LSTM<T>(50, 20))->gradClip(10),
            new Linear<T>(20, 1)
        ),
        sequenceLength
    );

    Nadam<T> optimizer(nn, new CriticSequencer<T>(new MSE<T>(false)));
    optimizer.learningRate(learningRate);

    Tensor<T> preds(tFeat.size(0), 1, 1);
    extrapolate(nn, feat3D, preds);

    T minErr = optimizer.critic().forward(preds, tLab3D), err;
    clog << "Initial error: " << minErr << endl;

    SequenceBatcher<T> batcher(feat, lab, sequenceLength, bats);
    Progress p(epochs, clog);
    p.display(0);

    for(size_t i = 0; i < epochs; ++i)
    {
        batcher.reset();
        nn.forget();

        optimizer.step(batcher.features(), batcher.labels());

        extrapolate(nn, feat3D, preds);
        err = optimizer.critic().forward(preds, tLab3D);
        if(err < minErr)
            minErr = err;

        p.display(i + 1);
        clog << "\terr: " << err << "\tmin: " << minErr << flush;
    }

    return 0;
}
