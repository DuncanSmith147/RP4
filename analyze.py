
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


from collections import defaultdict

import nltk

from . import extract
from . import utils
from . import tokenize

from imp import reload
reload(extract)
reload(utils)
reload(tokenize)

def eval_indices(indices):
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

def check_rule(words, rule):
    # words is a list of words from a section / paragraph / sentence
    # each word must be separated from punctuation characters e.g. ['dog', '.']
    # rule is a list of sets of keywords
    # be careful with case (e.g. lower case all words)
    indices = [[] for _ in rule]
    for i, word_set in enumerate(rule):
        intersect = word_set.intersection(words)
        if intersect:
            indices[i].extend([words.index(word) for word in intersect])
        else:
            return False
    return eval_indices(indices)

def naive(filename, pipeline, rules_file):
    # get text as single string and lower case
    text = extract.process(filename).decode().lower()
    # apply pipeline
    word_lists = list(pipeline(text))
    # get rules
    rules_data = utils.read_json_data(rules_file)
    # for efficiency remove longer descriptions and
    # convert lists to sets
    rules = {}
    for key, lis in rules_data.items():
        rules[key] = [set(item) for item in lis[1:]]
    # apply each rule to each list of words
    res = defaultdict(list)
    for key, rule in rules.items():
        for word_list in word_lists:
            if check_rule(word_list, rule):
                res[key].append(word_list)
    return res


