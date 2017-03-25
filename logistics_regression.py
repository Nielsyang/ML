import numpy as np
import os
class logistic_regression(object):
    def __init__(self, feature_num, epoch=20, batch_size=128, learning_rate=0.05,
                 dataset='default', epsilon=1e-7, beta1=0.9, beta2=0.995):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.feature_num = feature_num
        self.epoch = epoch
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    def forward(self, data, w, b):
        return 1-self.sigmoid(np.dot(data, w)+b)

    def get_batch_data(self, idx):
        if idx+self.batch_size > self.train_size:
            np.random.shuffle(self.train_data)
            return self.train_data[:self.batch_size,:-1], self.train_data[:self.batch_size,-1], self.batch_size
        else:
            return self.train_data[idx:idx+self.batch_size,:-1], self.train_data[idx:idx+self.batch_size,-1], idx+self.batch_size

    def backward(self,data,label):
        """use Adam optimizer"""
        self.logits = self.forward(data, self.w, self.b)
        for i in xrange(self.feature_num):
            dw = 0
            for j in xrange(self.batch_size):
                dw += -data[j][i]*(self.logits[j]-label[j])
            # gradient check
            # e = 1e-4
            # w1 = np.array([self.w[0]+e, self.w[1]])
            # w2 = np.array([self.w[0]-e, self.w[1]])
            # print (self.cost(self.batch_data, self.batch_label, w1, self.b)-self.cost(self.batch_data, self.batch_label, w2, self.b))/(2*e)
            self.momentum_w[i] = self.beta1*self.momentum_w[i] + (1-self.beta1)*dw
            self.v_w[i] = self.beta2*self.v_w[i] + (1-self.beta2)*(dw**2)
            self.w[i] += -self.learning_rate*self.momentum_w[i]/(np.sqrt(self.v_w[i])+self.epsilon)

        db = 0
        for i in xrange(self.batch_size):
            db += -(self.logits[j]-label[j])
        self.v_b = self.beta2*self.v_b + (1-self.beta2)*(db**2)
        self.momentum_b = self.beta1*self.momentum_b + (1-self.beta1)*db
        self.b += -self.learning_rate*self.momentum_b/(np.sqrt(self.v_b)+self.epsilon)

    def cost(self, x, y, w, b):
        loss = 0
        for i in xrange(self.batch_size):
            loss += y[i]*np.log(self.forward(x,w,b)[i]) + (1-y[i])*np.log(1-self.forward(x,w,b)[i])
        return loss

    def train_model(self):
        train_file = os.path.join('.\\data', self.dataset, 'train.txt')
        self.train_data = []
        self.train_label = []
        with open(train_file, 'r') as f:
            for line in f.readlines():
                self.train_data.append(line[:-1].split(','))

        self.train_data = np.array(self.train_data).astype(np.float64)
        self.train_size = len(self.train_data)
        self.w = np.random.randn(self.feature_num,)
        self.b = 0
        self.momentum_w = np.zeros_like(self.w)
        self.momentum_b = 0
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0
        idx = 0
        for i in xrange(self.epoch*self.train_size/self.batch_size):
            self.batch_data,self.batch_label,idx = self.get_batch_data(idx)
            self.backward(self.batch_data, self.batch_label)
            accuracy = 0
            for j in xrange(self.batch_size):
                if (self.logits[j] > 0.5 and self.batch_label[j] == 1) or (self.logits[j] < 0.5 and self.batch_label[j] == 0):
                    accuracy += 1
            print 'Mini batch accuracy: %.2f' % (accuracy*1.0/self.batch_size)
        print self.w, self.b

if __name__ == '__main__':
    lr = logistic_regression(feature_num=2, dataset='fake_data')
    lr.train_model()
