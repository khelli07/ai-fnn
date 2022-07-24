class Accuracy:
    @staticmethod
    def calculate(ypred, ytrue):
        return (ytrue == ypred).sum() / len(ytrue)

    def name():
        return "accuracy"
