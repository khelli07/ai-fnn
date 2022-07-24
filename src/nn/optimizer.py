class SGD:
    def __init__(self, learning_rate=0.005):
        self.lr = learning_rate

    def generate_nabla(self, delta, prev_activation):
        nabla_w = self.lr * (delta @ prev_activation.T)
        nabla_b = self.lr * (delta)
        return nabla_w, nabla_b
