# models.py

from optimizers import *
from nerdata import *
from utils import *

import random
import time

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        decodedViaViterbi = HmmNerModel.viterbi(self,sentence_tokens)
        return decodedViaViterbi

    def viterbi(self, sentence_tokens: List[Token]) -> LabeledSentence:
        # raise Exception("IMPLEMENT ME")
        # implementing viterbi decoding here
        # 1.)Initial
        # for each state s, calculate score1(s)=P(s)P(x1|s)=pisBx1,s
        # 2.)Recurrence
        # for i = 2 to n, for every state s, calculate
        # scorei(s)=maxP(s|yi-1)P(xi|s)scorei-1(yi-1)
        # 3.) Final State
        # calculate maxP(y,x|pi,A,B)=maxscoren(S)

        # first thing I got to do is figure out what the states are
        # the states are the tags?
        lastTag = None
        numStates = len(self.tag_indexer)  # tag_indexer is a bijection between objects and ints for tags
        numSentenceTokens = len(sentence_tokens)
        scores = np.zeros((numStates, numSentenceTokens))
        s = np.zeros(numStates)
        # backpointers = np.zeros((numSentenceTokens, numStates))
        # maybe use a list or dictionary instead for backpointers
        # print(self.emission_log_probs)
        backpointers = []  # list not working well, f1 @ 7ish
        backpointers = {}  # dictionary not working f1 went from 6ish to 0.0 , try np array next ?
        predictionTags = []  # make it a list like it was in BadNerPy

        # step 1
        token = sentence_tokens[0]
        # Token(Rotor, NNP, I-NP) example token
        # Rotor example word from token
        currentWord = token.word
        currentWordIndex = self.word_indexer.index_of(currentWord)
        for state in range(numStates):
            # transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
            # init_log_probs: [num_tags]-length array containing initial sequence log probabilities
            scores[state, 0] = self.init_log_probs[state] + self.emission_log_probs[state, currentWordIndex]
        # backpointers[0] = self.tag_indexer.get_object(0)
        # step 2
        for i in range(1, numSentenceTokens):
            currentWord = sentence_tokens[i].word
            currentWordIndex = self.word_indexer.index_of(currentWord)
            if currentWordIndex == -1:
                # word_indexer.add_and_get_index("UNK") from trainHMMModel
                currentWordIndex = self.word_indexer.index_of("UNK")  # word is unknown because Indexer didn't find it
            for state in range(numStates):
                for lastState in range(numStates):
                    #   #get max of transitions + emissions + prevScore
                    # A yi-1,s
                    transitions = self.transition_log_probs[lastState, state]
                    # B s,xi
                    emissions = self.emission_log_probs[state, currentWordIndex]
                    # score yi -1 = A+B+score i-1 (yi-1)
                    s[lastState] = transitions + emissions + scores[lastState, i - 1]
                    # print(s[lastState])
                    # s[lastState] is giving me negative numbers. Am I making a mistake ???
                # get max of transitions + emissions + prevScore
                # score at i = max of A+B+score i-1(yi-1)
                scores[state, i] = np.max(s)
                # STEP 3
                # Final Step: Calculate
                # max score n of s
                finalState = np.argmax(s)
                # the backpointer dictionary will map a key (i and state) to a value
                # the key can't just be the index because there are mulitple states at each i
                # list implementation failng here
                backpointers[(str(i), str(state))] = finalState
                # lastTag = np.argmax()
                s = np.zeros(numStates)
            # print("FINAL STATES!!")
            # print(finalState)
            #  backpointers.append(finalState)
            # backpointers[(i-1,state)] = self.tag_indexer.get_object(np.argmax(s))

        # gets the max of the scores of every state
        # Once you run your dynamic program, you still need to extract the best answer. Typically this is done
        # by either storing a backpointer for each cell to know how that cell’s value was derived or by using a
        # backward pass over the chart to reconstruct the sequence.
        # follwing code is from BadNerPy try to use it somehow ?
        # pred_tags = []
        # for tok in sentence_tokens:
        #    if tok.word in self.words_to_tag_counters:
        #       # [0] selects the top most common (tag, count) pair, the next [0] picks out the tag itself
        #      pred_tags.append(self.words_to_tag_counters[tok.word].most_common(1)[0][0])
        #  else:
        #      pred_tags.append("O")
        # print(scores)
        # print("prediction tags")
        # print(predictionTags)
        # predictionTagObjects=[]
        # print(scores.shape)
        #
        #   tagObj = self.tag_indexer.get_object(backpointers[i])
        # predictionTags.append(backpointers[i])
        # for k,v in backpointers:
        # print(v)
        # predictionTags.append(v)
        # predictionTagObjects.append()
        # predictionTags.append([0)
        # for index in range(len(sentence_tokens)):
        #  predictionTags.append()
        # predictionTags.append(self.tag_indexer.get_object(np.argmax(scores[:,-1])))
        # predictionTags.append(np.argmax(scores[:,-1]))
        # index = len(sentence_tokens) - 1
        #  while index > 0:
        # print(backpointers[index,predictionTags[-1]])
        #     index-=1

        # predictionTags.append(np.argmax(scores[:,-1]))
        # index = len(sentence_tokens) - 1
        #    while index > 0:
        # print(backpointers[index,predictionTags[-1]])
        #     predictionTags.append(backpointers[(index, predictionTags[-1])])
        #    index-=1
        # index = 0
        # currentSpot = len(sentence_tokens) - 1
        # print(predictionTags)
        # while index < len(sentence_tokens):
        #    predictionTags[index] = self.tag_indexer.get_object(predictionTags[currentSpot-index])
        #   print(predictionTags[index])
        #  index+=1
        # currentSpot-=1
        # print(predictionTags)

        # predictionTags.append(np.argmax(scores[:,-1]))

        predictionTags.append(np.argmax(scores[:, -1]))
        tagObjects = [None] * (numSentenceTokens + 1)
        index = numSentenceTokens - 1
        temp = predictionTags[0]
        tagObjects[0] = self.tag_indexer.get_object(temp)
        objects = []
        while index > 0:
            predictionTags.append(backpointers[(str(index), str(predictionTags[-1]))])
            # tagObjects[index] = self.tag_indexer.get_object(predictionTags[numSentenceTokens-1-index])
            index -= 1
        index = numSentenceTokens - 1
        while index >= 0:
            objects.append(self.tag_indexer.get_object(predictionTags[index]))
            index -= 1
        # tagObjects[0] = self.tag_indexer.get_object(temp)
        # Labeled Sentence takes a list of tokens and a list of chunks
        # chunks_from_bio_tag_seq is form BadNerPy and converts biotags to (start,end,label)
        # def viterbi(self, init_log_probs, transition_log_probs, emission_log_probs):
        #    return 1
        # print(tagObjects)
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(objects))

    def forwardbackwordAlgo(sentence, tag_indexer, scorer, feature_indexer):
        #not implemented :(
        #forward part returns alpha
        alpha = HmmNerModel.forward(sentence, tag_indexer, scorer, feature_indexer)
        #backword part returns beta
        beta = HmmNerModel.forward(sentence, tag_indexer, scorer, feature_indexer)
        return (alpha,beta)

    def forward(sentence, tag_indexer, scorer, feature_indexer):
        return 1
    def backward(sentence, tag_indexer, scorer, feature_indexer):
        return 1

def train_hmm_model(sentences: List[LabeledSentence], silent: bool=False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    if not silent:
        print("Tag indexer: %s" % tag_indexer)
        print("Initial state log probabilities: %s" % init_counts)
        print("Transition log probabilities: %s" % transition_counts)
        print("Emission log probs too big to print...")
        print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
        print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
        print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


##################
# CRF code follows

class FeatureBasedSequenceScorer(object):
    """
    Feature-based sequence scoring model. Note that this scorer is instantiated *for every example*: it contains
    the feature cache used for that example.
    """
    def __init__(self, tag_indexer, feature_weights, feat_cache):
        self.tag_indexer = tag_indexer
        self.feature_weights = feature_weights
        self.feat_cache = feat_cache

    def score_init(self, sentence, tag_idx):
        if isI(self.tag_indexer.get_object(tag_idx)):
            return -1000
        else:
            return 0

    def score_transition(self, sentence_tokens, prev_tag_idx, curr_tag_idx):
        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
        if (isO(prev_tag) and isI(curr_tag))\
                or (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) \
                or (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
            return -1000
        else:
            return 0

    def score_emission(self, sentence_tokens, tag_idx, word_posn):
        feats = self.feat_cache[word_posn][tag_idx]
        return self.feature_weights.score(feats)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        raise Exception("IMPLEMENT ME")


def train_crf_model(sentences: List[LabeledSentence], silent: bool=False) -> CrfNerModel:
    """
    Trains a CRF NER model on the given corpus of sentences.
    :param sentences: The training data
    :param silent: True to suppress output, false to print certain debugging outputs
    :return: The CrfNerModel, which is primarily a wrapper around the tag + feature indexers as well as weights
    """
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    if not silent:
        print("Training")
    weight_vector = UnregularizedAdagradTrainer(np.zeros((len(feature_indexer))), eta=1.0)
    num_epochs = 3
    random.seed(0)
    for epoch in range(0, num_epochs):
        epoch_start = time.time()
        if not silent:
            print("Epoch %i" % epoch)
        sent_indices = [i for i in range(0, len(sentences))]
        random.shuffle(sent_indices)
        total_obj = 0.0
        for counter, i in enumerate(sent_indices):
            if counter % 100 == 0 and not silent:
                print("Ex %i/%i" % (counter, len(sentences)))
            scorer = FeatureBasedSequenceScorer(tag_indexer, weight_vector, feature_cache[i])
            (gold_log_prob, gradient) = compute_gradient(sentences[i], tag_indexer, scorer, feature_indexer)
            total_obj += gold_log_prob
            weight_vector.apply_gradient_update(gradient, 1)
        if not silent:
            print("Objective for epoch: %.2f in time %.2f" % (total_obj, time.time() - epoch_start))
    return CrfNerModel(tag_indexer, feature_indexer, weight_vector)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)


def compute_gradient(sentence: LabeledSentence, tag_indexer: Indexer, scorer: FeatureBasedSequenceScorer, feature_indexer: Indexer) -> (float, Counter):
    """
    Computes the gradient of the given example (sentence). The bulk of this code will be computing marginals via
    forward-backward: you should first compute these marginals, then accumulate the gradient based on the log
    probabilities.
    :param sentence: The LabeledSentence of the current example
    :param tag_indexer: The Indexer of the tags
    :param scorer: FeatureBasedSequenceScorer is a scoring model that wraps the weight vector and which also contains a
    feat_cache field that will be useful when computing the gradient.
    :param feature_indexer: The Indexer of the features
    :return: A tuple of two items. The first is the log probability of the correct sequence, which corresponds to the
    training objective. This value is only needed for printing, so technically you do not *need* to return it, but it
    will probably be useful to compute for debugging purposes.
    The second value is a Counter containing the gradient -- this is a sparse map from indices (features)
    to weights (gradient values).
    """
    x = HmmNerModel.forwardbackwordAlgo(sentence, tag_indexer, scorer, feature_indexer)
    gradient = x # x should be an accumulation of what we get from x
    return gradient
    raise Exception("IMPLEMENT ME)")