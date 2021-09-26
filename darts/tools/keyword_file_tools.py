import numpy as np
from darts.physics import value_vector
import os.path as osp
import re


def get_table_keyword(file_name, keyword):
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip() == keyword:
                table = []
                while True:
                    row = f.readline()
                    if row[0] == '#':
                        continue
                    else:
                        a = np.fromstring(row.strip(), dtype=np.float, sep=' ')
                    if a.size > 0:
                        table.append(value_vector(a))
                    if row.find('/') != -1:
                        return table
                break


def load_single_keyword(file_name, keyword, def_len=1000, cache=0):
    read_data_mode = 0
    pos = 0
    cache_filename = file_name + '.' + keyword + '.cache'

    if cache:
        # if caching is enabled and cache file is already created, read from it
        import os
        if os.path.isfile(cache_filename):
            print("Reading %s from %s..." % (keyword, cache_filename), end='', flush=True)
            a = np.fromfile(cache_filename)
            print(" %d values have been read." % len(a))
            return a

    # start with specified (or default) array length
    a = np.zeros(def_len)
    with open(file_name, 'r') as f:
        for line in f:
            s_line = line.strip()

            # requested keyword is not yet detected
            if read_data_mode == 0:
                # to support PETREL files read the first word in the line (there could be comments after the keyword
                # in the same line)
                first_word = s_line.split(maxsplit=1)
                # check if the line is not empty, if so - take the first word
                if first_word:
                    first_word = first_word[0]

                if first_word == keyword:
                    # requested keyword is now detected
                    read_data_mode = 1
                    print("Reading %s from %s..." % (keyword, osp.abspath(file_name)), end='', flush=True)
                    continue
                if s_line == 'INCLUDE':
                    path = osp.abspath(osp.dirname(file_name))
                    include = osp.join(path, f.readline().strip(' \\/\n'))
                    a = load_single_keyword(include, keyword, def_len)
                    if a.size > 0:
                        return a
                    else:
                        continue
            # requested keyword is not yet detected or comment found - skip the line
            if not read_data_mode or len(s_line) == 0 or s_line[0] == '#':
                continue
            # collect all float values to numpy array
            # check for repeating values
            if s_line.find('*') != -1:
                b = []
                s1 = s_line.split()
                for x in range(s1.__len__()):
                    if s1[x].find('*') != -1:
                        s2 = s1[x].split('*')
                        s2_add = np.ones(int(s2[0]), dtype=float)
                        s2_add.fill(s2[1])
                        b = np.append(b, s2_add)
                    else:

                        try:
                            value = float(s1[x])
                        except ValueError:
                            # in PETREL the trailing slash can be on the same line with numbers
                            # Skip the message if that is the case
                            if s1[x] != '/':
                                print("\n''", s1[x],  "'' is not a float, skipping...\n")
                            continue
                        b = np.append(b, value)
            else:
                b = np.fromstring(s_line, dtype=np.float, sep=' ')

            # Check if there is still enough place in array
            # if not, enlarge array by a factor of 2
            while pos + b.size > def_len:
                def_len *= 2
                a.resize(def_len, refcheck=False)
            # copy data from b to a
            a[pos:pos + b.size] = b
            pos += b.size

            # break when slash found
            if line.find('/') != -1:
                break
    # shrink the array to actual read length
    a.resize(pos, refcheck=False)


    if cache:
        # if caching is enabled, save to cache file
        a.tofile(cache_filename)
        print(" %d values have been read and cached." % pos)
    else:
        print(" %d values have been read." % pos)

    return a

def save_few_keywords(fname, keys, data):
    f = open(fname, 'w')
    for id in range(len(keys)):
        f.write(keys[id])
        for i, val in enumerate(data[id]):
            if i % 4 == 0: f.write('\n')
            f.write("%12.10f" % val)
            f.write('\t')
        f.write('\n' + '/' + '\n')
    f.close()
