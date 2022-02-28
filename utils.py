
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

# various functions / classes that don't obviously belong elsewhere

from pathlib import Path
import json

from nltk.corpus import wordnet as wn


"""Find synonyms and hyponyms via WordNet"""

def synonyms(synset, as_names=True):
    # Returns the synonyms (lemmas) for synset.
    # Returns strings if as_names is True.
    if as_names:
        # The .split('.')[0] is not necessary below, but is included
        # in case the API is changed to make lemme name consistent with
        # synset names
        return [' '.join(lemma.name().split('.')[0].lower().split('_'))
                for lemma in synset.lemmas()]
    return synset.lemmas()

def hyponyms(synset, as_names=True):
    # Returns the hyponyms for synset.
    # Returns strings if as_names is True.
    if as_names:
        return [' '.join(hyponym.name().split('.')[0].lower().split('_'))
                for hyponym in synset.hyponyms()]
    return synset.hyponyms()

def all_words_from_rules(rules_data):
    # This function is specific to the rules
    # defined in "Code Sheet.doc.x" (with minor adjustments).
    # A rule is a mapping of a key to a list containing a string
    # (longer description) and two lists, the first containing
    # terms that will often be present, and the second containing
    # terms that will often follow the terms in the first list.
    # The function simply extracts all the words in the two lists across
    # all the rules.
    words = set()
    for key, data in rules_data.items():
        for lis in data[1:]: # we don't want the description
            words.update(lis)
    return words

def create_syndict(words):
    # Create mapping of words to synonyms.
    # "words" is an iterable containing strings and / or
    # Synset instances.
    # For strings the synsets are generated and the first is
    # taken as the relevant sense of the word.
    # The return value is a mapping (dict) of words to a list
    # containing a string (word / phrase definition" and a list
    # of synonyms.
    # In some cases the first in the list of synsets will not
    # be the correct sense (hence the inclusion of the definition)
    # and the relevant synset will need to be found and
    # substituted for the word in a subsequent function call.
    syndict = {}
    for word in set(words):
        if isinstance(word, str):
            synsets = wn.synsets(word)
            if synsets:
                synset = synsets[0] # try first item
                syndict[word] = [synset.definition(), synonyms(synset)]
            else:
                # word / phrase not in WordNet
                syndict[word] = ['', [word]]
        else:
            # word is a synset
            synset = word
            word = ' '.join(synset.name().split('.')[0].lower().split('_'))
            syndict[word] = [synset.definition(), synonyms(synset)]
    return syndict

def create_hypdict(words):
    # Create mapping of words to hyponyms (derived terms).
    # Other aspects of this function are identical to
    # those of create_syndict.
    hypdict = {}
    for word in set(words):
        if isinstance(word, str):
            synsets = wn.synsets(word)
            if synsets:
                synset = synsets[0] # try first item
                hypdict[word] = [synset.definition(), hyponyms(synset)]
            else:
                # word / phrase not in WordNet
                hypdict[word] = ['', [word]]
        else:
            # word is a synset
            synset = word
            word = ' '.join(synset.name().split('.')[0].lower().split('_'))
            hypdict[word] = [synset.definition(), hyponyms(synset)]
    return hypdict

def reverse_map(dic):
    # Create a mapping (dict) of synonyms / hyponyms to words.
    # "dic" is mapping of the kind returned by a call to
    # either "create_syndict" or "create_hypdict".
    # Can be used to harmonize text data.
    reverse_map = {}
    for word, (definition, values) in dic.items():
        for val in values:
            reverse_map[val] = word
    return reverse_map

def read_json_data(filename):
    # Read data from a json formatted file.
    # If filename is not a valid path the
    # file is assumed to be in the same directory as this module.
    if filename == Path(filename).name:
        # assume file is in the same directory as module
        filename = Path(__file__).parent / filename
    with open(filename, 'r') as f:
        return json.load(f)

def write_json_data(data, filename):
    # Write data to a text file in json format.
    # If filename is not a valid path the
    # file will be placed in the same directory as this module.
    if filename == Path(filename).name:
        # assume file is in the same directory as module
        filename = Path(__file__).parent / filename
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
        

"""Utility function for chaining generators"""

class Pipeline(list):
    # Concatenates generators into an object
    # that behaves like a single generator.
    # >>> p_line = Pipeline([lambda x: (y**2 for y in x), lambda x: (y+2 for y in x)])
    # >>> list(p_line([0,1,2,3]))
    # [2, 3, 6, 11]
    def __call__(self, arg):
        for item in self:
            arg = item(arg)
        return arg


"""Function for replacing words according to supplied mapping"""

def find_replace(words, mapping):
    # "words" is a list of strings.
    # "mapping" is dict mapping words to
    # the words that will replace them
    # in the returned list.
    return [mapping.get(word, word) for word in words]
