# LADAN
The source code of article "Distinguish Confusing Law Articles for Legal Judgment Prediction", ACL 2020

## Statement
Due to the space limitation of our Dropout, here we only can provide the processed dataset of CAIL-small. After the dual anonymity period ends, we will open source data and code on github.com.

## Usage
The "__training_code__" folder contains the training code of our LADAN. And the "__data_and_config__" folder contains the deta preprocessing codes and processed datasets. 
Here are details of some important folders in the "__data_and_config__" folder:

* config: Records the experimental parameter setting.
* Model: Contains some basic models we use.
others: the data preprocessing codes.
	
For training, you should move all folders in the "data_and_config" folder into "training_code" folder first, i.e.,<br> 

	cp -r data_and_config/ training_code/

Then, go to "training_code" folder and train any model that starts with "LADAN+". An example likes: <br>

	cd training_code
	python LADAN+Topjudge_small.py --config CONFIG_FILE --gpu GPU_ID
	
where "__CONFIG_FILE__" corredponds to path of files in the "config" folder and the gpu config is necessary.
