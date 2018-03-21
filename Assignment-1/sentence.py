#!/usr/bin/env python3
import re
import sys

filename = sys.argv[1]
f = open(filename)
corpus = f.read()

sent = re.sub("([^GKC])(\.|\?|!|\])('?)(\s*|$)('?)([A-Z0-9])",  "\g<1>\g<2>\g<3></s>\g<4><s>\g<5>\g<6>", corpus)
#print(sent)

sent2 = re.sub("Mr\.</s>(\s)<s>", "Mr.\g<1>", sent)
#print(sent2)

final = "<s>"+sent2[:-1]+"</s>"
final = re.sub("\.(\n+)", ".</s>\g<1><s>", final)
print(final)
