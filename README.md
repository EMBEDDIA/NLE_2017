# Code for paper 'Combining bag-of-words and deep convolutional features for language variety classification' #

## Installation, documentation ##

Python 3 is needed.<br/>
Clone the project from the repository with 'git clone http://source.ijs.si/mmartinc/NLE_2017'<br/>
Install dependencies if needed: pip3 install -r requirements.txt

To reproduce the results published in the paper run the code in the command line using following commands:

python3 train.py


You can also use the system on your own custom datasets:<br/>
Following arguments are available:

--data_directory : Path to data directory<br/>
--train_corpus : Path to train corpus - first column should be text, second a label. Columns should be separated by tab.<br/>
--dev_corpus : Path to development corpus - first column should be text, second a label. Columns should be separated by tab..<br/>
--test_corpus: Path to test corpus - first column should be text, second a label. Columns should be separated by tab.

For further costumization, you can tweak the code.


## Contributors to the code ##

Matej Martinc<br/>
Senja Pollak

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
