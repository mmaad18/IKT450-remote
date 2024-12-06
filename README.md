# IKT450-remote
Code written for the IKT450 project at UiA

## Commands for remote server setup

### Create directories

```bash
mkdir ~/Documents
mkdir ~/Documents/IKT450
mkdir ~/Documents/Datasets
mkdir ~/Documents/Datasets/Fish_GT
```

### Get Fish Ground Truth dataset

```bash
cd ~/Documents/Datasets/Fish_GT
wget https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/class_id.csv 
wget https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar 
tar -xvf fishRecognition_GT.tar
```


