
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


import nltk

from . import extract
from . import utils


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

def naive(filename):
    # get text as string
    text = extract.process(filename).decode()
    # sentence tokenize
    sents = nltk.sent_tokenize(text)
    # word tokenize
    sents = [nltk.word_tokenize(sent) for sent in sents]
    # let's try word replacement to our standard synonyms / hyponyms
    syndict = utils.read_json_data('synonyms.dat')
    hypdict = utils.read_json_data('hyponyms.dat')
    sents = [utils.find_replace(sent, hypdict) for sent in sents]
    sents = [utils.find_replace(sent, syndict) for sent in sents]
    # apply rules
    rules_data = utils.read_json_data('rules.dat')
    # for efficiency convert remove longer descriptions and
    # convert lists to sets
    rules = {}
    res = []
    for key, lis in rules_data.items():
        rules[key] = [set(item) for item in lis[1:]]
    #return rules
    for sent in sents:
        for key, lis in rules.items():
            indices = [[] for _ in lis]
            for i, word_set in enumerate(lis):
                intersect = word_set.intersection(sent)
                if intersect:
                    indices[i].extend([sent.index(word) for word in intersect])
                else:
                    break
            else:
                if eval_indices(indices):
                    res.append(key)
    return res

    
