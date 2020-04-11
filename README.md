# LADAN
The source code of article "Distinguish Confusing Law Articles for Legal Judgment Prediction", ACL 2020

## Statement
	Due to the space limitation of our Dropout, here we only can provide the processed dataset of CAIL-small. After the dual anonymity period ends, we will open source data and code on github.com.
## Usage
	The "training_code" folder contains the training code of our LADAN. And the "data_and_config" folder contains the deta preprocessing codes and processed datasets. 
	Here are details of some important folders in the "data_and_config" folder:
	
	"leal_basis_data": Contains the peocessed datasets of CAIL-small, including train set "train_processed_thulac_Legal_basis.pkl", valid set "valid_processed_thulac_Legal_basis.pkl" and test set "test_processed_thulac_Legal_basis.pkl"
	"config": Records the experimental parameter setting.
	"Model": Contains some basic models we use.
	others: the data preprocessing codes.
	
	For training, you should move all folders in the "data_and_config" folder into "training_code" folder first, i.e.,
		cp -r data_and_config/ training_code/
	Then, go to "training_code" folder and train any model that starts with "LADAN+". An example likes:
		cd training_code
		python LADAN+Topjudge_small.py --config CONFIG_FILE --gpu GPU_ID
	where "CONFIG_FILE" corredponds to path of files in the "config" folder and the gpu config is necessary.
