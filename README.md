Comment Spam Classification
=========

This is a weekend project that aims to remove spammed comments from the food reviews at zomato.com. The comment data used for training and testing the classifier was obtained from Zomato.

This can also be used as a generic text classifier(commments/ reviews/ articles).

After much trial and testing over the given data, this was the ideal approach taken:
- Form bag of words, bigrams and trigrams of the text content.
- Based on the type of data given, extract structural features. *For example, if html is present in content, find the frequency count of each tag etc.*

The code can be easily modified to add and remove features and also tune the classifiers.

>*Before running this script, you should install numpy, scipy, nltk, enchant, scikit learn and Beautiful Soup (if html is present)*

 