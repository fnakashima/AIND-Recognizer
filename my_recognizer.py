import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # https://discussions.udacity.com/t/recognizer-implementation/234793/5
    # for word, model in models.items():
    #     calculate the scores for each model(word) and update the 'probabilities' list.
    #     determine the maximum score for each model(word).
    #     Append the corresponding word (the tested word is deemed to be the word for which with the model was trained) to the list 'guesses'.

    hwords = test_set.get_all_Xlengths()
    for word_id in range(0,len(hwords)):
        probs = {}
        best_score = float('-inf')
        best_word = None
        X, lengths = hwords[word_id]

        for word, model in models.items():
            # Tip: The hmmlearn library may not be able to train or score all models. 
            # Implement try/except contructs as necessary to eliminate non-viable models from consideration
            # https://discussions.udacity.com/t/failure-in-recognizer-unit-tests/240082/3
            logL = float('-inf')
            try:
                logL = model.score(X, lengths)
            except:
                pass

            probs[word] = logL

            if logL > best_score:
                best_score = logL
                best_word = word
        
        probabilities.append(probs)
        guesses.append(best_word)

    return probabilities, guesses
