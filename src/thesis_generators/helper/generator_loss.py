
class GeneratorLoss():

    def __init__(self, model, generator, possibility_estimator, lmb) -> None:
        pass

    def dandl_loss(true_seq, cf_seq, true_outcome, cf_outcome, distance_function, **kwargs):
        pass

    def wachter_loss(true_seq, cf_seq, true_outcome, cf_outcome, distance_function, **kwargs):
        pass


class GeneratorDistance():
    def multivariate_distance(self, true_emb, cf_emb, true_seq, cf_seq):
        # Maybe with sequence alignment method
        # Or longest common sequence method
        # Or multivariate ARIMA/ARMA -> https://www.quora.com/Whats-the-difference-between-ARMA-ARIMA-and-ARIMAX-in-laymans-terms-What-exactly-do-P-D-Q-mean-and-how-do-you-know-what-to-put-in-for-them-in-say-R-1-0-2-or-2-1-1
        pass

class GeneratorMetric():
    def levenshstein(true_seq, cf_seq, true_outcome, cf_outcome, distance_function, **kwargs):
        pass

    def damerau_levenshtein(true_seq, cf_seq, true_outcome, cf_outcome, distance_function, **kwargs):
        pass