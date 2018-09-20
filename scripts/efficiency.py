#!/usr/bin/python

import sys

ndocs = 52343021
docs = range(ndocs)

d = open('/data/suwaileh/clueweb12/b/la3/doc-id-map/clueweb12_catb.docs.labels')

i = 0
for l in d:
    docs[i] = l.split()[0]
    i += 1

d.close()

fname = sys.argv[1]
gname = fname + '.qq'

f = open(fname)
g = open(gname, 'w')

for q in range(50):
    for k in range(1000):
        l = f.readline().split()
        l[2] = docs[int(l[2]) - 1]
        l = ' '.join(l)
        g.write(l + '\n')

f.close()
g.close()
