# ML2019Spring-Final-DeepQ

## Dataset
you should put data in github folder, it should look like:

    models
    src
    train(folder include train image)
    test(folder include test image)
    train_labels.csv
    report.pdf
    requirement.txt
    .
    .
    .

## Pretrained model
If you want to reproduce kaggle result, run this script to download pretrained model

    bash ./get_model.sh
 The shell script will download the models for both yolov1 and retinanet
 
 ## Yolov1 case
 
 ### Predict by pretrained model
 This shell script will load pretraind model 'yolo.pth' and generate submission.csv for kaggle 
 
    bash ./yolo_test.sh
 | kaggle public score | kaggle private score |
 | :--: | :--: |
 | 0.21330 | 0.17911 |
 ### Train own model
You can simply run this script to train model, this use first 17500 images for train and the rest for validation and save model in folder models 

    bash ./yolo_train.sh
#### Date preprocessing
You can change line 39-42 in yolo_split.py to split data
#### Predict by own model
Change model path in yolo_test.py (line 38)

## Retinanet case

### Predict by pretrained model
This shell script will load pretraind model 'yolo.pth' and generate submission.csv for kaggle
