# Code for paper 'Combining n-grams and deep convolutional features for language variety classification' #

## Installation, documentation ##

Python 3 is needed.<br/>
Clone the project from the repository with 'git clone http://source.ijs.si/mmartinc/NLE_2017'<br/>
Install dependencies if needed: pip3 install -r requirements.txt

To reproduce the results published in the paper run the code in the command line using following commands:

python3 train.py --experiment DSLCC;<br/>
python3 train.py --experiment GDIC;<br/>
python3 train.py --experiment ADIC;<br/>

The DSLCC v4.0 and ADIC data can be found in the data folder. The GDIC corpus is not available but you can ask the Vardial 2018 GDI task administrator for a copy and download
it in the /data/gdic folder.

You can also use the system on your own custom datasets. Following arguments are available:

--experiment: Default is DSLCC. Other allowed values are: GDIC, ADIC and OTH (for custom datasets).<br/>
--data_directory : Path to data directory<br/>
--train_corpus : Path to train corpus - first column should be text, second a label. Columns should be separated by tab.<br/>
--dev_corpus : Path to development corpus - first column should be text, second a label. Columns should be separated by tab..<br/>
--test_corpus: Path to test corpus - first column should be text, second a label. Columns should be separated by tab.
--weighting: Define the weighting scheme, it can either be "tfidf" or "bm25". Default is "tfidf"

For further costumization, just tweak the code or send me an email (matej.martinc@ijs.si) :).


## Contributors to the code ##

Matej Martinc<br/>
Senja Pollak

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
