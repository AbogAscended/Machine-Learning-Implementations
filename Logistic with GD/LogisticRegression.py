import torch as th

def sigmoid(x):
    return 1 / (1 + th.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate=.2, threshold=1e-5):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        self.weights = None
        self.threshold = threshold
        self.bias = None
        self.X = None
        self.y = None
        self.N = None

    def fit(self, x, y):
        self.X = th.tensor(x,device=self.device,dtype=th.float64)
        self.N = x.shape[0]
        self.y = th.tensor(y,device=self.device,dtype=th.float64)
        n_samples, n_features = x.shape
        scale = (2.0 / (n_features + 1))**.5
        self.weights = th.randn([n_features, 1], device=self.device, dtype=th.float64) * scale
        self.bias = th.tensor(0.0,device=self.device,dtype=th.float64)

    def train(self):
        loss = [th.tensor(1e10,device=self.device), th.tensor(1e9,device=self.device)]
        print(f"Training with learning rate {self.lr}")
        while loss[-1]-loss[-2] < 0 and th.abs(loss[-1]-loss[-2]) > self.threshold:
            z = th.matmul(self.X, self.weights) + self.bias
            logistic = sigmoid(z)
            delta_x = logistic - self.y

            gradient_w = (1/self.N) * th.matmul(self.X.T, delta_x)
            gradient_b = (1/self.N) * th.sum(delta_x)

            self.weights = self.weights - gradient_w * self.lr
            self.bias = self.bias - gradient_b * self.lr

            eps = 1e-11
            l1 = self.y * th.log(logistic)
            l2 = (1 - self.y) * th.log(1 - logistic + eps)
            loss_total = l1 + l2
            loss.append(-1*loss_total.mean())
            if (len(loss)-2) <= 10: print(f'step {len(loss)-2}: {loss[-1].item()}')
            elif (len(loss)-2)%100 == 0: print(f'step {len(loss)-2}: {loss[-1].item()}')
        print(f'Training finished with lr: {self.lr} and threshold: {self.threshold}')

    def predict(self, x, threshold=0.5):
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, x_test):
        x_test_tensor = th.tensor(x_test, device=self.device, dtype=th.float64)
        z = th.matmul(x_test_tensor, self.weights) + self.bias
        probabilities = sigmoid(z)

        return probabilities.cpu().numpy()
