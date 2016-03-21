#! /usr/bin/python

__author__="Daniel Bauer <bauer@cs.columbia.edu>"
__date__ ="$Sep 12, 2011"

import sys
from collections import defaultdict
import math

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()
        self.pi_table = {}
        self.pi_table[(0, '*','*')] = 1

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in xrange(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()


        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count


    def get_emission_parameter(self, word, tag):
        emission_count = self.emission_counts[(word, tag)]
        tag_count = self.ngram_counts[0][(tag,)]
        return emission_count / tag_count

    def make_rare_category(self):
        converted_emission_counts = defaultdict(int)
        rare = {}
        for key, count in self.emission_counts.items():
            if count < 5:
                tag = key[1]
                if tag in rare:
                    rare[tag] += count
                else:
                    rare[tag] = count
            else:
                converted_emission_counts[key] = count

        for tag, count in rare.items():
            converted_emission_counts[('_RARE_', tag)] = count

        self.emission_counts = converted_emission_counts

    def get_most_likely_tag_baseline(self, word):
        most_likely = (None, 0)
        for tag in self.all_states:
            e = self.get_emission_parameter(word, tag)
            if e > most_likely[1]:
                most_likely = (tag, e)

        if not most_likely[0]:
            for tag in self.all_states:
                e = self.get_emission_parameter('_RARE_', tag)
                if e > most_likely[1]:
                    most_likely = (tag, e)

        return most_likely[0]


    def tag_input_baseline(self, input, output):
        tagged = []
        for line in input:
            word = line.strip()
            if word == '':
                tagged.append('')
            else:    
                tag = self.get_most_likely_tag_baseline(word)
                tagged.append(word + ' ' + tag)

        output.write('\n'.join(tagged))

    def tag_input_viterbi(self, input, output):
        tagged = []

        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(input)), self.n)
        index = 1

        tag_at_index_minus_1 = '*'
        tag_at_index_minus_2 = '*'

        for complex_trigram in ngram_iterator:
            print complex_trigram
            word = complex_trigram[2][1]
            description = complex_trigram[2][0]
            if description == None:
                continue
            else:
                tag = self.get_most_likely_tag_viterbi(index, tag_at_index_minus_2, tag_at_index_minus_1, word)[1]
                tagged.append((word, tag))
                tag_at_index_minus_2 = tag_at_index_minus_1
                tag_at_index_minus_1 = tag
                index += 1

        # output.write('\n'.join(tagged))

    def get_most_likely_tag_viterbi(self, index, tag_at_index_minus_2, tag_at_index_minus_1, word):
        max_prob = 0
        max_prob_tag = None

        for tag in self.all_states:
            e = self.get_emission_parameter(word, tag)
            trigram = (tag_at_index_minus_2, tag_at_index_minus_1, tag)
            q = self.get_q_param(trigram)
            pi = self.pi_table[(index - 1, tag_at_index_minus_2, tag_at_index_minus_1)]
            prob = e * q * pi
            self.pi_table[(index, tag_at_index_minus_1, tag)] = prob
            if prob > max_prob:
                max_prob = prob
                max_prob_tag = tag

        return (max_prob, max_prob_tag)

    def get_q_param(self, trigram):
        trigram_count = self.ngram_counts[2][trigram]
        bigram_count = self.ngram_counts[1][(trigram[0],trigram[1],)]
        return trigram_count / bigram_count


def usage():
    print """
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = file(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    # counter.train(input)
    # counter.write_counts(file('./gene.counts.clean','w'))
    counter.read_counts(input)
    counter.tag_input_viterbi(file('./gene.dev','r'), 'output')

    # i = file('./gene.dev', 'r')
    # o = file('./gene_dev.p1.out', 'w')
    # counter.tag_input(i, o);
