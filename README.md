# Transfer Learning in COVID-19 Detection 

## DD2424 Deep Learning in Data Science Group Project
## May 2020

### Authors
- Hannes Kindbom (hkindbom@kth.se)
- Ershad Taherifard (ershadt@kth.se)
- Mikael Ljung (milju@kth.se)
- Johanna Dyremark (dyremark@kth.se)

### The project and used dataset
This group project has aimed to examine the possibilities of using transfer learning in detecting COVID-19 as a way of overcoming the challenges in having a limited dataset consisting of COVID-19 X-ray scans. The implementation has been inspired by the paper "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images" and uses the same dataset as utilized in this paper with the exception of the validation dataset, which consists of COVID-19 X-Ray images that have been added after the first publication of the paper. These specification can be found in folder /dataset_specifications with scripts for processing and sorting the data in /preprocessing folder. Half of the experiments in the project have been performed on a subset of the available data to investigate the need for larger data amounts and the impact of imbalanced classes. 

### Running the code
To run the model.py file, the specific experiment (1, 2, 3 or 4) should be entered as a command line argument. The dataset should be placed into /data/dataset/largeDataset or /data/dataset/smallDataset, firstly sorted into train, val and test folders and secondly into their separate class folders in order for ImageDataGenerator() to work. 
