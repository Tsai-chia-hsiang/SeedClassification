### Kaggle Competition: Plant Seedlings Classification

https://www.kaggle.com/competitions/plant-seedlings-classification/overview

third-party package :
- numpy 
- pandas 
- matplotlib
- torch, torchvision

** More detail are available at ```requirements.txt```


## Method:
- using pretrianed __torchvision.model.resnet50__ as the based model to do transfer learning by fine-tuning its ```fc``` layer.

## Exectuion step:
- setup.py:

    - To download pytorch pretrained ResNet50 model 
    
    - split the images in ```data/train/``` into training data and test data, then store the splitting result.

        - The split result: 
            <img src="./data/trainvalloader/TrainValcount/TrainValcount.jpg">
            
- trainmodel.py:

    To train the transfered ResNet50 model by modifing its fully connected layers (i.e. classifier)

    the log and the model will be store at ```model/transferRN50_id/``` .

- testmodel.py:

    To generate the ```submission.csv``` for testing images in ```data/test/```