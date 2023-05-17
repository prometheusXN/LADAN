# LADAN
The source code of article "Distinguish Confusing Law Articles for Legal Judgment Prediction", ACL 2020.

Besides, the Tensorflow 2.x version is avaliable at our [**D-LADAN**](https://github.com/prometheusXN/D-LADAN) opject.

## Data_processing 
When you get the CAIL datasets, run '__data_and_config/data/tongji3.py__' to get '__{}_cs.json__' first.
Then choose the corresponding code among '__data_processed/data_pickle.py__', '__data_processed_big/data_pickle_big.py__', and '__data/make_Legal_basis_data.py__' to generate the data structure according to model you want to run.

You can get the word embeddin file "__cail_thulac.npy__" at the following address: https://drive.google.com/file/d/1_j1yYuG1VSblMuMCZrqrL0AtxadFUaXC/view?usp=drivesdk+
the dataset file "__CAIL2018.zip__" at the address: https://drive.google.com/file/d/12QOsAumyzmdsqgqhqcNOhOaGcp--ECnm/view?usp=sharing, 
https://drive.google.com/file/d/1JsP6co2GiCodv4oqyjaa6Q8TY5FWOQDM/view?usp=sharing and 
https://drive.google.com/file/d/1VX8YhqF6ZX2tVLeOJV4VxdnNkWEDrbyS/view?usp=sharing

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

References
-----------

BibTex:

```Latex
@inproceedings{xu-etal-2020-distinguish,
    title = {Distinguish Confusing Law Articles for Legal Judgment Prediction},
    author = {Xu, Nuo  and
      Wang, Pinghui  and
      Chen, Long  and
      Pan, Li  and
      Wang, Xiaoyan  and
      Zhao, Junzhou},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year = {2020},
    doi = {10.18653/v1/2020.acl-main.280},
    pages = {3086--3095}
}
```


