
##Copyright (c) 2022 duncan g. smith
##
##Permission is hereby granted, free of charge, to any person obtaining a
##copy of this software and associated documentation files (the "Software"),
##to deal in the Software without restriction, including without limitation
##the rights to use, copy, modify, merge, publish, distribute, sublicense,
##and/or sell copies of the Software, and to permit persons to whom the
##Software is furnished to do so, subject to the following conditions:
##
##The above copyright notice and this permission notice shall be included
##in all copies or substantial portions of the Software.
##
##THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
##OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
##THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
##OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
##ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
##OTHER DEALINGS IN THE SOFTWARE.


import os
from collections import defaultdict, Counter
import json

import nltk
import pandas as pd

from . import extract
from . import utils
from . import tokenize

##from imp import reload
##reload(extract)
##reload(utils)
##reload(tokenize)

def _eval_indices(indices):
    # "indices" is a list of lists of integers (indices).
    # returns True if it is possible to choose an index from
    # each of the lists in turn s.t. the selected indices are
    # in increasing order.
    minval = -1
    for inds in indices:
        inds = iter(inds)
        cand = None
        for ind in inds:
            if ind > minval:
                cand = ind
                break
        for ind in inds:
            if minval < ind < cand:
                cand = ind
        if cand is None:
            return False
        minval = cand
    return True

def _eval_indices2(indices):
    # "indices" is a list of lists of integers (indices).
    # returns the number of ways it is possible to choose an index from
    # each of the lists in turn s.t. the selected indices are
    # in increasing order.
    indices = [sorted(lis) for lis in indices]
    # go through process of constructing tree
    # but just record leaf nodes and frequencies
    # (numpaths) at each level
    last_indices = indices[-1]
    last_degrees = [1]*len(indices[-1])
    for inds in reversed(indices[:-1]):
        #print ('called', inds, last_indices)
        nodes, degrees = _f(inds, last_indices)
        if not nodes:
            break
        #print (nodes, degrees)
        #print ('ld', last_degrees)
        last_degrees = last_degrees[-degrees[0]:]
        #print ('ld', last_degrees)
        # accumulate last_degrees in reverse
        for i in range(-2, -len(last_degrees)-1, -1):
            last_degrees[i] += last_degrees[i+1]
        #print ('ld', last_degrees)
        last_degrees = [last_degrees[len(last_degrees) - degree] for degree in degrees]
        #print ('ld', last_degrees)
        last_indices = nodes
    else:
        return sum(last_degrees)
    return 0

def _eval_indices2_(indices):
    # brute force approach for testing purposes
    # checks each element in the Cartesian product
    # for monotonicity
    from itertools import product
    indices = [sorted(lis) for lis in indices]
    cnt = 0
    for tup in product(*indices):
        if all(v<w for v, w in zip(tup, tup[1:])):
            cnt += 1
    return cnt

def _f(lis1, lis2):
    # lis1 and lis2 are sorted lists of
    # objects that support comparison
    # returns the nodes (with non-zero outdegree)
    # and the degrees of
    # the bipartite graph with an edge from
    # each node, v, in lis1 to each node, w,
    # in lis2 s.t. v < w
    it1 = iter(lis1)
    it2 = enumerate(lis2)
    nodes = []
    degrees = []
    v = next(it1)
    j, w = next(it2)
    try:
        while True:
            if v < w:
                nodes.append(v)
                degrees.append(len(lis2) - j)
                v = next(it1)
            else:
                j, w = next(it2)
    except StopIteration:
        pass
    return nodes, degrees

def _test_eval_indices2():
    import random
    for _ in range(100):
        data = [random.choices(range(30), k=6),
                random.choices(range(40), k=6),
                random.choices(range(50), k=6),
                random.choices(range(60), k=6),
                random.choices(range(70), k=6),
                random.choices(range(80), k=6)]
        if not _eval_indices2(data) == _eval_indices2_(data):
            print ('Fail')

def check_rule(words, rule):
    # words is a list of words from a section / paragraph / sentence
    # each word must be separated from punctuation characters e.g. ['dog', '.']
    # rule is a list of sets of keywords
    # be careful with case (e.g. lower case all words)
    indices = [[] for _ in rule]
    for i, word_set in enumerate(rule):
        for j, word in enumerate(words):
            if word in word_set:
                indices[i].append(j)
        if not indices[i]:
            return False
    return _eval_indices(indices)

def check_freq(words, rule):
    indices = [[] for _ in rule]
    for i, word_set in enumerate(rule):
        for j, word in enumerate(words):
            if word in word_set:
                indices[i].append(j)
        if not indices[i]:
            return 0
    return _eval_indices2(indices)

def get_sentence_texts(filename):
    # A convenience function that illustrates
    # how to construct a pipeline and use it
    # to generate sentence texts
    # Returns a list
    # get raw text
    text = extract.process(filename).decode().lower()
    # construct pipeline
    pipeline = utils.Pipeline([nltk.sent_tokenize, tokenize.word_tokenize])
    filter1 = utils.replacement_factory(utils.read_json_data('synonyms.dat'))
    filter2 = utils.replacement_factory(utils.read_json_data('hyponyms.dat'))
    pipeline.extend([filter1, filter2])
    # process raw data and return
    # list of texts
    return list(pipeline(text))

def get_paragraph_texts(filename):
    # A convenience function that illustrates
    # how to construct a pipeline and use it
    # to generate paragraph texts
    # Returns a list
    # get raw text
    text = extract.process(filename).decode().lower()
    # construct pipeline
    pipeline = utils.Pipeline([tokenize.paragraph_tokenize, tokenize.word_tokenize])
    filter1 = utils.replacement_factory(utils.read_json_data('synonyms.dat'))
    filter2 = utils.replacement_factory(utils.read_json_data('hyponyms.dat'))
    pipeline.extend([filter1, filter2])
    # process raw data and return
    # list of texts
    return list(pipeline(text))


class Analyzer:
    def __init__(self, rules):
        # if rules is a filepath load
        # rules (assumed to be in json format)
        if isinstance(rules, (str, bytes, os.PathLike)):
            self.rules = utils.read_json_data(rules)
        else:
            self.rules = rules

    def analyze(self, texts):
        raise NotImplementedError


class BoolAnalyzer(Analyzer):
    def __init__(self, rules):
        Analyzer.__init__(self, rules)
        # for efficiency remove longer descriptions and
        # convert lists to sets
        for key, lis in list(self.rules.items()):
            self.rules[key] = [set(item) for item in lis[1:]]
        self.texts = None
        self.results = None

    def analyze(self, texts):
        # list of lists of words
        res = defaultdict(list)
        for i, text in enumerate(texts):
            for key, rule in self.rules.items():
                if check_rule(text, rule):
                    # always appends 1, but
                    # store anyway as derived
                    # classes might have different scores
                    res[key].append((i, 1))
        self.texts, self.results = texts, res

    def table_output(self):
        # returns array with rows corresponding
        # to indices of texts and columns corresponding
        # to rule keys (in sorted order)
        # Returns a sparse matrix (dict mapping coordinates
        # to non-zero values) with text indicess as row indices
        # and sorted feature keys as columns
        if self.results is None:
            raise ValueError("No results to output")
        texts, res = self.texts, self.results
        keys = sorted(self.rules.keys())
        arr = {}
        for j, key in enumerate(keys):
            for i, score in res[key]:
                arr[(i,j)] = score
        return arr

    def data_frame(self, strings=None, suppress=False):
        if self.results is None:
            raise ValueError("No results to output")
        texts, res = self.texts, self.results
        keys = sorted(self.rules.keys())
        # create dictionary of sparse arrays with
        # insertion order equal to order of keys (i.e. sorted)
        d = {}
        for key in keys:
            lis = [0]*len(texts)
            for i, j in res[key]:
                lis[i] = j
            d[key] = pd.arrays.SparseArray(lis, dtype='int32')
        frame = pd.DataFrame(d, columns=keys)
        if strings:
            if not len(strings) == len(texts):
                raise ValueError(("'strings' has length {} while "
                                  "analysis is based on {} 'texts'").format(len(strings), len(texts)))
            frame.insert(loc=0, column='Text', value=strings)
        if suppress:
            if strings:
                frame = frame.iloc[[i for i in range(frame.shape[0]) if frame.values[i,1:].sum()],
                                   [j for j in range(frame.shape[1]) if j==0 or frame.values[:,j].sum()]]
            else:
                frame = frame.iloc[[i for i in range(frame.shape[0]) if frame.values[i,:].sum()],
                                   [j for j in range(frame.shape[1]) if frame.values[:,j].sum()]]
        return frame


class FreqAnalyzer(BoolAnalyzer):
    # uses same rules so inherit from BoolAnalyzer
    def __init__(self, rules):
        BoolAnalyzer.__init__(self, rules)

    def analyze(self, texts):
        # list of lists of words
        res = defaultdict(list)
        for i, text in enumerate(texts):
            for key, rule in self.rules.items():
                freq = check_freq(text, rule)
                if freq:
                    res[key].append((i, freq))
        self.texts, self.results = texts, res


class HybridAnalyzer(FreqAnalyzer):
    # inherit from FreqAnalyzer because only the analysis differs
    def __init__(self, rules):
        FreqAnalyzer.__init__(self, rules)

    def analyze(self, texts):
        # list of lists of words
        res = defaultdict(list)
        for i, text in enumerate(texts):
            for key, rule in self.rules.items():
                score = 2*check_freq(text, rule)
                if not score:
                    cntr = Counter(text)
                    for keyword in rule[0]:
                        score += cntr[keyword]
                if score:
                    res[key].append((i, score))
        self.texts, self.results = texts, res


def to_latex(frames, filename, landscape=False, **kwargs):
    # Export the data frames to a LateX file.
    # Places one table on each page.
    # Optionally choose landscape.
    # Automatically scales tables to fit
    # to page (if too large).
    # Using some keyword arguments for 'frame.to_latex'
    # will produce a LaTeX file that will not
    # compile without errors.
    # In this case the file generated must be edited.
    with open(filename, 'w') as f:
        f.write(r'\documentclass[a4paper]{article}')
        f.write('\n\n')
        f.write('\n'.join([r'\usepackage{booktabs}',
                           r'\usepackage{adjustbox}']))
        f.write('\n')
        if landscape:
            f.write(r'\usepackage[landscape, margin=0.5in]{geometry}')
        else:
            f.write(r'\usepackage[margin=0.5in]{geometry}')
        f.write('\n\n')
        f.write(r'\begin{document}')
        f.write('\n\n')
        f.write(r'\centering')
        for i, frame in enumerate(frames):
            if not i == 0:
                f.write('\n\n')
                f.write(r'\newpage')
            f.write('\n\n')
            f.write(r'\begin{adjustbox}{max width=\linewidth, max totalheight=\textheight}')
            f.write('\n\n')
            f.write(frame.to_latex(**kwargs))
            f.write('\n')
            f.write(r'\end{adjustbox}')
        f.write('\n\n')
        f.write(r'\end{document}')

    


