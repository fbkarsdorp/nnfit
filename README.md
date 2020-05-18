# Classifying Evolutionary Forces in Languages Change

A fundamental problem in research into language and cultural change is the difficulty of
distinguishing processes of stochastic drift (also known as neutral evolution) from
processes that are subject to certain selection pressures. In this article, we describe a
new technique based on Deep Neural Networks, in which we reformulate the detection of
evolutionary forces in cultural change as a binary classification task. Using Residual
Networks for time series trained on artificially generated samples of cultural change, we
demonstrate that this technique is able to efficiently, accurately and consistently learn
which aspects of the time series are distinctive for drift and selection. We compare the
model with a recently proposed statistical test, the Frequency Increment Test, and show
that the neural time series classification system provides a possible solution to some of
the key problems of this test.

## Data

Code to reconstruct the past-tense data set can be obtained from
https://github.com/mnewberry/ldrift. To run the past-tense analysis in
`notebooks/past-tense.ipynb`, save the frequency list under `data/coha-past-tense.txt`. 

## Requirements
All code is implemented in Python 3.7. A detailed list of the requirements to run the code
can be found in the `requirements.txt` file.

## Training

To train your own models, run `src/train.py` and follow the instructions therein. 

---
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

