
import numpy as np
import collections
import tree as treeM

class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,1))

    # TODO initialization
    def initParams(self):

        # Params
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.dWs = np.empty(self.Ws.shape)

    def costAndGrad(self,mbdata): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W
           Gradient w.r.t. L in sparse form.
        """
        cost = 0.0

        # Zero gradients
        self.dW[:] = 0
        self.dWs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
           cost += forwardProp(tree.root)

        # Back prop each tree in minibatch
        for tree in mbdata:
            backProp(tree.root)
        
        return [self.dW,self.dWs,self.dL]

    def forwardProp(self,node):
        cost = 0.0

        if node.isLeaf:
            node.hActs = self.L[:,node.word]
            node.fprop = True

        else:
            if not node.left.fprop: 
                cost += self.forwardProp(node.left)
            if not node.right.fprop:
                cost += self.forwardProp(node.right)
            # Affine
            node.hActs = np.dot(self.W,
                    np.hstack([node.left.hActs, node.right.hActs])) + self.b
            # Relu
            node.hActs[node.hActs<0] = 0

        # Softmax
        node.probs = np.dot(self.Ws,node.hActs) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)

        node.fprop = True

        # Cost TODO nansum here
        return cost - np.log(node.probs[node.label])

    def backProp(self,node,error=None):

        # Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
#        self.dbs += deltas
        deltas = np.dot(self.Ws.T,deltas)
        self.dWs += deltas
        
        if not node.isRoot:
            deltas += error

        # Leaf nodes update word vecs
        if node.isLeaf:
            dL[node.word] += deltas
            return

        # Hidden grad
        if not node.isLeaf:
            self.dW += np.dot(deltas,
                    np.hstack([node.left.acts, node.right.acts]).T)
            # Error signal to children
            deltas = np.dot(self.W.T, deltas) * (node.hActs!=0)
            backProp(node.left, deltas[:self.wvecDim])
            backProp(node.right, deltas[self.wvecDim:])
        
   
if __name__ == '__main__':

    train = treeM.loadTrain()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    rnn = RNN(wvecDim,outputDim,numW,mbSize=1)
    rnn.initParams()

    mbData = [train[0]]






