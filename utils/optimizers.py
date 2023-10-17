import numpy as np

class SGD:
    def __init__(self, lr=0.01, decay=0, momentum=0, regularization=None, lambd=0):
        self.lr = lr
        self.decay = decay  # 0 for no decay
        self.momentum = momentum 
        self.iterations = 0 
        self.s_w = None
        self.s_b = None 

        self.regularization = regularization
        self.lambd = lambd

    def regularization_term(self, weights):
        if self.regularization == 'l2':
            return [self.lambd * w for w in weights]
        elif self.regularization == 'l1':
            return [self.lambd * np.sign(w) for w in weights]
        else:
            return [np.zeros_like(w) for w in weights]

    def update(self, weights, biases, grads):
            self.iterations += 1 
            # calculate decayed lr
            lr_t = self.lr * (1. / (1. + self.decay * self.iterations))  

            reg_term = self.regularization_term(weights)

            # initialize velocities if they don't exist yet
            if self.s_w is None:
                self.s_w = [np.zeros_like(w) for w in weights]
                self.s_b = [np.zeros_like(b) for b in biases]

            for i in range(len(weights)):
                # calculate velocity updates for momentum
                if self.momentum > 0:
                    self.s_w[i] = self.momentum * self.s_w[i] - lr_t * (grads[f'W{i+1}'] + reg_term[i])
                    self.s_b[i] = self.momentum * self.s_b[i] - lr_t * grads[f'b{i+1}']  
                    weights[i] += self.s_w[i]
                    biases[i] += self.s_b[i]
                else:
                    # standard SGD update (without momentum)
                    weights[i] -= lr_t * (grads[f'W{i+1}'] + reg_term[i]) 
                    biases[i] -= lr_t * grads[f'b{i+1}']

            return weights, biases
    

class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., weight_decay=0.01):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay 
        self.weight_decay = weight_decay 
        self.iterations = 0
        self.m_w = None  # first moment vector
        self.s_w = None  # second moment vector
        self.m_b = None
        self.s_b = None

    def update(self, weights, biases, grads):
        self.iterations += 1
        lr_t = self.lr * (1. / (1. + self.decay * self.iterations))

        # update weights according to the equation in slide 30 of:
        # https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf

        # init
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.s_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.s_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            # moving avg of gradients (momentum)
            self.m_w[i] = self.beta_1 * self.m_w[i] + (1. - self.beta_1) * grads[f'W{i+1}']
            self.m_b[i] = self.beta_1 * self.m_b[i] + (1. - self.beta_1) * grads[f'b{i+1}']

            # bias-corrected first moment estimate
            m_w_hat = self.m_w[i] / (1. - self.beta_1 ** self.iterations)
            m_b_hat = self.m_b[i] / (1. - self.beta_1 ** self.iterations)

            # moving avg of the squared gradients (RMSprop)
            self.s_w[i] = self.beta_2 * self.s_w[i] + (1. - self.beta_2) * (grads[f'W{i+1}'] ** 2)
            self.s_b[i] = self.beta_2 * self.s_b[i] + (1. - self.beta_2) * (grads[f'b{i+1}'] ** 2)

            # bias-corrected second raw moment estimate
            s_w_hat = self.s_w[i] / (1. - self.beta_2 ** self.iterations)
            s_b_hat = self.s_b[i] / (1. - self.beta_2 ** self.iterations)

            # update params
            weights[i] -= lr_t * m_w_hat / (np.sqrt(s_w_hat) + self.epsilon) + (self.weight_decay * weights[i])
            biases[i] -= lr_t * m_b_hat / (np.sqrt(s_b_hat) + self.epsilon)

        return weights, biases
