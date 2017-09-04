import math
import statistics
import warnings
import random

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def get_BIC_score(self, num_states, aplpha):
        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            
        logL = model.score(self.X, self.lengths)
        # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/6
        # https://discussions.udacity.com/t/verifing-bic-calculation/246165/4
        # where d is the number of features and n the number of components.
        # n = num_states
        # d = len(self.X[0])
        # n_parameters = n * n + 2 * n * d - 1

        # From the reviewer's suggestion:
        # An alternative way to calculate p directly from the HMM model
        # p (n_parameters) is the sum of four terms:
        # - The free transition probability parameters, which is the size of the transmat matrix
        # - The free starting probabilities
        # - Number of means
        # - Number of covariances which is the size of the covars matrix
        #
        # These can all be calculated directly from the hmmlearn model as follows:
        # p = (model.startprob_.size - 1) + (model.transmat_.size - 1) + model.means_.size + model.covars_.diagonal().size
        # http://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm
        # https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py#L106
        n_parameters = (model.transmat_.size - 1) \
                     + (model.startprob_.size - 1) \
                     + model.means_.size \
                     + model.covars_.diagonal().size

        # https://discussions.udacity.com/t/number-of-data-points-bic-calculation/235294
        n_dataPoints = len(self.X)
        return -2 * logL + n_parameters * np.log(n_dataPoints) * aplpha

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        best_num_components = self.min_n_components
        best_score = float('inf')
        num_states = self.min_n_components

        # From the reviewer's suggestion:
        # You can add a parameter alpha to the free parameters to provide a weight to the free parameters, so the penalty term will become alpha * p * logN.
        # This regularization parameter can then be varied to further improve the result 👍🏽. Good values can be anywhere between 0 and 1, or even greater than 1 (but maybe less than 2 :) )
        alpha = random.uniform(0,1)
        while num_states <= self.max_n_components:
            try:
                BIC_score = self.get_BIC_score(num_states, alpha)
            except ValueError:
                num_states += 1
                continue

            # Model selection: The lower the BIC value the better the model
            if best_score > BIC_score:
                best_score = BIC_score
                best_num_components = num_states
            
            num_states += 1

        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def get_DIC_score(self, num_states):
        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        
        # A difference between the likelihood of the data, log(P(X(i)), 
        #  and the average of anti-likelihood terms, log(P(X(all but i),
        #  where the anti-likelihood of the data Xj against model M is a likelihood-like quantity
        #  in which the data and the model belong to competing categories.
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

        # https://discussions.udacity.com/t/dic-score-calculation/238907
        # log(P(X(i)) is simply the log likelyhood (score) that is returned from the model by calling model.score.
        logL = model.score(self.X, self.lengths)

        # The log(P(X(j)); where j != i is just the model score
        #  when evaluating the model on all words other than the word for which we are training this particular model.
        logLs_anti = []
        for antiX, antiLength in self.hwords.items():
            if antiX == self.this_word: continue
            try:
                logLs_anti.append(model.score(antiX, antiLength))
            except:
                pass

        #average_logL_anti = sum(logLs_anti) / (len(self.hwords)-1)
        average_logL_anti = np.mean(sum(logLs_anti))

        # From the reviewer's suggestion
        # The calculation of the DIC penalty term can be done in one line only as follows:
        # average_logL_anti = np.mean([model.score(self.hwords[word]) for word in self.words if word != self.this_word])
        # However, all the result states were 2 with this one line formula.
        # Probably, try/except could be necessary here to catch an error caused by any words which cannot be calculated a score as above.

        return logL - average_logL_anti

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # implement model selection based on DIC scores
        best_num_components = self.min_n_components
        best_score = float('-inf')
        num_states = self.min_n_components
        while num_states <= self.max_n_components:
            try:
                DIC_score = self.get_DIC_score(num_states)
            except ValueError:
                num_states += 1
                continue

            # select the configuration that yields the highest value of the model selection criterion.
            if best_score < DIC_score:
                best_score = DIC_score
                best_num_components = num_states
            
            num_states += 1

        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def get_LogL_with_CV(self, split_method, num_states):
        split_logLs = []
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            # train a model with train fold
            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
            model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
            # Cross-validation with test fold
            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
            split_logLs.append(model.score(test_X, test_lengths))
            
        # return the average of all the splits
        return np.mean(split_logLs)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV

        # https://discussions.udacity.com/t/cannot-have-number-of-splits-n-splits-3-greater-than-the-number-of-samples-2/248850/5
        # https://discussions.udacity.com/t/fish-word-with-selectorcv-problem/233475
        # Take min value between length of sequences and 3 to set n_splits for KFold
        # Otherwise, get the following error
        # ValueError: Cannot have number of splits n_splits=3 greater than the number of samples: 2.
        n_splits = min(len(self.sequences), 3)

        # If n_splits is 1, just returns model with n_constant
        # Otherwise, get the following error when splitting by KFold
        # "ValueError: k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1."
        # https://discussions.udacity.com/t/problems-in-part-3-executing-selectorcv-valueerror/302848/2
        if n_splits < 2:
            return self.base_model(self.n_constant)

        split_method = KFold(n_splits=n_splits)

        best_num_components = self.min_n_components
        best_score = float('-inf')
        num_states = self.min_n_components
        while num_states <= self.max_n_components:
            try:
                logL_with_CV = self.get_LogL_with_CV(split_method, num_states)
            except ValueError:
                num_states += 1
                continue

            # Compare the result with the best score so far
            if best_score < logL_with_CV:
                best_score = logL_with_CV
                best_num_components = num_states
            
            num_states += 1

        return self.base_model(best_num_components)
