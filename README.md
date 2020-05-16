# Transfer Learning in COVID-19 Detection 

## DD2424 Deep Learning in Data Science Group Project
## May 2020

### Authors
- Hannes Kindbom (hkindbom@kth.se)
- Ershad Taherifard (ershadt@kth.se)
- Mikael Ljung (milju@kth.se)
- Johanna Dyremark (dyremark@kth.se)

### The project and used dataset
This group project has aimed to examine the possibilities of using transfer learning in detectin COVID-19 as a way to overcome the challenge of having a limited limited dataset consisting of COVID-19 X-ray scans. The implementation has been inspired by the paper and uses the same dataset as utilized in the referenced paper with the exception of the validation dataset, which consists of COVID-19 X-Ray images which have been added after the first publication of the paper. These specification can be found in folder "preprocessing". Half of the experiments in the project have been performed on a subset of the available data to investigate the need for larger data amounts and the impact of imbalanced classes. 

### Running the code
To run the model.py file, the specific experiment (1, 2, 3 or 4) should be entered as a command line argument. The dataset should be placed into /data/dataset/largeDataset or /data/dataset/smallDataset, firstly sorted into train, val and test folders and secondly into their separate class folders in order for ImageDataGenerator() to work. 
