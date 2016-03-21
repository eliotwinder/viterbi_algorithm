#! /usr/bin/python
#

import sys

def scrubber(input_path, output_path):
    input = file(input_path, 'r')
    output = file(output_path, 'w')
    words = {}
    
    for line in input:
        if line != '\n':
          [word, tag] = line.split(' ')
          if word in words:
              words[word] += 1
          else:
              words[word] = 1

    cleaned = []

    input = file(input_path, 'r')

    for line in input:
        if line != '\n':
            [word, tag] = line.split(' ')
            if words[word] < 5:
                cleaned.append('_RARE_' + ' ' + tag)
            else:
                cleaned.append(word + ' ' + tag)
        else:
            cleaned.append('\n')

    output.write(''.join(cleaned))
        

if __name__ == "__main__":
    scrubber(sys.argv[1], sys.argv[2])
