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

void load(const std::string &infile, double validationPart, size_t &seriesColumn, Tensor<T> &feat, Tensor<T> &lab, Tensor<T> &tFeat, Tensor<T> &tLab, T &min, T &max)
{
    Tensor<T> series = Tensor<T>(FileSerializer::read(infile));
    seriesColumn = std::max(seriesColumn, series.size(1));
    min = math::min(series.narrow(1, seriesColumn));
    max = math::max(series.narrow(1, seriesColumn));
    normalize(series.narrow(1, seriesColumn));

    size_t trainLength = (1.0 - validationPart) * series.size(0);
    size_t testLength = series.size(0) - trainLength + 1;

    Tensor<T> train = series.narrow(0, 0, trainLength);
    Tensor<T> test = series.narrow(0, trainLength - 1, testLength);

    feat = train.narrow(0, 0, trainLength - 1);
    lab = train.narrow(0, 1, trainLength - 1).narrow(1, seriesColumn);
    tFeat = test.narrow(0, 0, testLength - 1);
    tLab = test.narrow(0, 1, testLength - 1).narrow(1, seriesColumn);
}

void extrapolate(Sequencer<T> &model, const Tensor<T> &context, const Tensor<T> &future, Tensor<T> &preds)
{
    model.forget();
    model.forward(context);

    preds.resize(future.size(0), 1, 1);
    for(size_t i = 0; i < future.size(0); ++i)
    {
        Tensor<T> inp(context.size(2));
        if(context.size(2) > 1)
            inp.view(context.size(2) - 1).copy(future.narrow(0, i));
        inp.narrow(0, context.size(2) - 1, 1).copy(model.output());
        preds.narrow(0, i).copy(model.forward(inp.view(1, 1, inp.size())));
    }
}

int main(int argc, const char **argv)
{
    ArgsParser args;
    args.addInt('b', "batchSize", 20);
    args.addInt('c', "seriesColumn", 0);
    args.addDouble('d', "learningRateDecay", 0.999);
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
    size_t seriesColumn   = args.getInt('c');
    size_t epochs         = args.getInt('e');
    size_t hiddenSize     = args.getInt('n');
    size_t sequenceLength = args.getInt('s');
    T learningRateDecay   = args.getDouble('d');
    T learningRate        = args.getDouble('l');
    T validationPart      = args.getDouble('v');
    string infile         = args.getString("infile");
    int seed              = args.getInt("seed");

    if(seed >= 0)
        RandomEngine::sharedEngine().seed(seed);

    Tensor<T> feat, lab, tFeat, tLab;
    T min, max;
    load(infile, validationPart, seriesColumn, feat, lab, tFeat, tLab, min, max);

    Tensor<T> feat3D  = feat.view(feat.size(0), 1, feat.size(1));
    Tensor<T> lab3D   = lab.view(lab.size(0), 1, lab.size(1));
    Tensor<T> tFeat3D = tFeat.view(tFeat.size(0), 1, tFeat.size(1));
    Tensor<T> tLab3D  = tLab.view(tLab.size(0), 1, tLab.size(1));

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

    Tensor<T> preds;
    extrapolate(nn, feat3D, tFeat3D, preds);

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
        optimizer.learningRate(optimizer.learningRate() * learningRateDecay);

        p.display(i + 1);

        extrapolate(nn, feat3D, tFeat3D, preds);
        err = optimizer.critic().forward(preds, tLab3D);
        if(err < minErr)
            minErr = err;
        clog << "\terr: " << err << "\tmin: " << minErr << flush;
    }

    extrapolate(nn, feat3D, tFeat3D, preds);
    clog << "Final error: " << optimizer.critic().forward(preds, tLab3D) << endl;

    return 0;
}
