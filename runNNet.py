import optparse
import cPickle as pickle

import sgd as optimizer
import rntn as nnet
import tree as tr
import time

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)

    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")


    (opts,args)=parser.parse_args(args)

    # Testing
    if opts.test:
        test(opts.inFile,opts.data)
        return
    
    print "Loading data..."
    # load training data
    trees = tr.loadTrees()
    opts.numWords = len(tr.loadWordMap())

    rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    rnn.initParams()

    sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)

    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d"%e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)
            rnn.toFile(fid)

def test(netFile,dataSet):
    trees = tr.loadTrees(dataSet)
    assert netFile is not None, "Must give model to test"
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        rnn.initParams()
        rnn.fromFile(fid)
    print "Testing..."
    cost,correct,total = rnn.costAndGrad(trees,test=True)
    print "Cost %f, Correct %d/%d, Acc %f"%(cost,correct,total,correct/float(total))


if __name__=='__main__':
    run()


