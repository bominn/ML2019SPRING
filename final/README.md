# ML2019Spring-Final-DeepQ

## Dataset
you should put data in github folder, it should look like:

    model
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
 
 ### Data preprocessing for training
This will generate train.txt and valid.txt

    python3 yolo_split.py
