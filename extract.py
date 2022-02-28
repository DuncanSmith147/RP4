
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

import textract


EXTENSION_SYNONYMS = {} # mapping of alternative extensions to the ones used here (lower case)


def process(filename, input_encoding=None, output_encoding='utf8',
            extension=None, **kwargs):
    """Passes arguments directly to textract.process initially. If the file
       extension (e.g. pdf) is not provided it is extracted from
       filename.
       Returns a bytestring encoded with 'output_encoding'.
       Supported filetypes and dependencies are listed at
       https://github.com/deanmalmgren/textract/blob/05fdc7a08dc3fc52eb519aefac4fcbec8981dd8e/docs/index.rst
       If textract does not support the extension, then an appropriate parser is sought in this module.
       If no suitable parser exists (in textract or this module) a NotImplementedError is raised.
    """
    try:
        txt = textract.process(filename, input_encoding=input_encoding,
                               output_encoding=output_encoding,
                               extension=extension, **kwargs)
    except textract.exceptions.ExtensionNotSupported:
        # see if it is supported in this module
        # get extension
        ext = extension
        if ext is None:
            _, ext = os.path.splitext(filename)
        else:
            if not ext.startswith('.'):
                ext = '.' + ext
        ext = ext.lower()
        if ext in EXTENSION_SYNONYMS:
            # get the extension that we use here
            ext = EXTENSION_SYNONYMS[ext]
        # look for relevant class or raise NotImplementedError if not found
        try:
            parser = globals()[ext[1:].capitalize() + 'Parser']()
        except KeyError:
            raise NotImplementedError('No available parser for extension {}'.format(extension))
        else:
            txt = parser.process(filename, input_encoding=input_encoding,
                                 output_encoding=output_encoding, **kwargs)
    return txt
    
# For convenience (and potentially contributing to textract) parsers can be derived
# from the following textract classes.
# Derived classes need only define an 'extract' method taking a filename as argument
# (and which is also passed **kwargs) that returns a byte-encoded or unicode string.
# ShellParser is the better option for parsers that need to run external programs
# See https://github.com/deanmalmgren/textract/tree/master/textract/parsers for details.

from textract.parsers import utils

BaseParser = utils.BaseParser
ShellParser = utils.ShellParser

# Place parser classes here
# For extension e.g. '.ext' the corresponding parser should be named 'ExtParser'

class TestParser(BaseParser):
    def extract(self, filename, **kwargs):
        # Read file and do whatever is required to
        # generate the text.
        # To test just call process with the path to a text
        # file and supply "extension='.test'".
        with open(filename, 'r') as f:
            return f.read()
