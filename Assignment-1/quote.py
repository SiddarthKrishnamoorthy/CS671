#!/usr/bin/env python3
import re

f = open("/home/siddarth/Documents/CS671A/Assignment-1/test.txt")
corpus = f.read()
right_quote = re.sub("'(\s|\.)", "\"\g<1>", corpus)
#print(right_quote)
left_quote = re.sub("(\A|\s)'", "\g<1>\"", right_quote)
#print(left_quote)
outlier1 = re.sub("\"Victoria(,?)(\"|')", "'Victoria\g<1>'", left_quote)
#print(outlier1)
outlier2 = re.sub("(I|you) (say|said) \"(.*)\"", "\g<1> \g<2> '\g<3>'", outlier1)
#print(outlier2)
outlier3 = re.sub("I (.*)?(called)(.*)?\"(.*)\"", "I \g<1>\g<2>\g<3>'\g<4>'", outlier2) # Blood
outlier4 = re.sub("I often said, \"(.*)\"", "I often said, '\g<1>'", outlier3) # Let the weak
outlier5 = re.sub("I cried out (.*) \"(.*)\"", "I cried out \g<1> '\g<2>'", outlier4)# cried out
outlier6 = re.sub("question, \"(.*)Victoria(.*)\"", "question, '\g<1>Victoria\g<2>'", outlier5)
print(outlier6)
