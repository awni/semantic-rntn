import numpy as np
import collections
np.seterr(over='raise',under='raise')

class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.
        """
        cost = 0.0
        correct = 0.0
        total = 0.0

        self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack
        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,corr,tot = self.forwardProp(tree.root)
            cost += c
            correct += corr
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.V**2)
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dV+self.rho*self.V),
                           scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node):
        cost = correct =  total = 0.0

        if node.isLeaf:
            node.hActs = self.L[:,node.word]
            node.fprop = True

        else:
            if not node.left.fprop: 
                c,corr,tot = self.forwardProp(node.left)
                cost += c
                correct += corr
                total += tot
            if not node.right.fprop:
                c,corr,tot = self.forwardProp(node.right)
                cost += c
                correct += corr
                total += tot
            # Affine
            lr = np.hstack([node.left.hActs, node.right.hActs])
            node.hActs = np.dot(self.W,lr) + self.b
            node.hActs += np.tensordot(self.V,np.outer(lr,lr),axes=([1,2],[0,1]))
            # Tanh
            node.hActs = np.tanh(node.hActs)

        # Softmax
        node.probs = np.dot(self.Ws,node.hActs) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)

        node.fprop = True

        return cost - np.log(node.probs[node.label]), correct + (np.argmax(node.probs)==node.label),total + 1


    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        # Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWs += np.outer(deltas,node.hActs)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T,deltas)
        
        if error is not None:
            deltas += error

        deltas *= (1-node.hActs**2)

        # Leaf nodes update word vecs
        if node.isLeaf:
            self.dL[node.word] += deltas
            return

        # Hidden grad
        if not node.isLeaf:
            lr = np.hstack([node.left.hActs, node.right.hActs])
            outer = np.outer(deltas,lr)
            self.dV += (np.outer(lr,lr)[...,None]*deltas).T
            self.dW += outer
            self.db += deltas
            # Error signal to children
            deltas = np.dot(self.W.T, deltas) 
            deltas += np.tensordot(self.V.transpose((0,2,1))+self.V,
                                   outer.T,axes=([1,0],[0,1]))
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)


if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    rntn = RNN(wvecDim,outputDim,numW,mbSize=4)
    rntn.initParams()

    mbData = train[:1]
    #cost, grad = rntn.costAndGrad(mbData)

    print "Numerical gradient check..."
    rntn.check_grad(mbData)






