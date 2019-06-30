# ML2019Spring-Final-DeepQ
>This is the final project from [ML2019SPRING](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html) which we are asked to implement object detection on medical image.  
>You can see the result in the [`report.pdf`](https://github.com/bominn/ML2019SPRING/blob/master/final/report.pdf)  
>[Kaggle link](https://www.kaggle.com/c/ml2019spring-final-deepq/overview)
## Toolkit version
> [`python`](https://www.python.org/): 3.6
> [`torch`](https://pytorch.org/): 1.0.1
> [`numpy`](http://www.numpy.org/): 1.16.2 
> [`pandas`](https://pandas.pydata.org/): 0.24.2
> [`torchvision`](https://pypi.org/project/torchvision/): 0.2.2
> [`cv2`](https://pypi.org/project/opencv-python/)
> [`PIL`](https://pypi.org/project/Pillow/): 5.4.1
> [`scikit-image`](https://scikit-image.org/)
> [`future`](https://pypi.org/project/future/):0.17.1  
> [The Python Standard Library](https://docs.python.org/3/library/)

# Usage:
To start with this project, please get the dataset first.
## Dataset
you should put train/test images and train_labels.csv in github folder, it should look like:

    models
    src
    train/
    ├── train00000.png
    ├── train00001.png .....
    test/
    ├── test0000.png
    ├── test0001.png .....
    train_labels.csv
    .gitignore
    README.md
    report.pdf
    requirement.txt
    .
    .
    .

## Pretrained model
If you want to reproduce kaggle result, run this script to download pretrained model first.

    bash ./get_model.sh
 The shell script will download the models for both yolov1 and retinanet. 
 If you are having trouble downloading the models using shell script, you can download the models from [yolo](https://drive.google.com/open?id=1eosYAAU1JnKwvGpczDHv1xfxG9lzarVz) and [retinanet](https://drive.google.com/open?id=1-gW-7ATWIdZ6bhfnRwIr7JKtTTjmy385) and put model in the folder `models/`.
 ## Yolov1 case
 
 ### Predict by pretrained model
 This shell script will load pretraind model `yolo.pth` and generate `submission.csv` for kaggle. 
 
    bash ./yolo_test.sh
 | kaggle public score | kaggle private score |
 | :--: | :--: |
 | 0.21330 | 0.17911 |
 ### Train your own model
You can simply run this script to train model, this use first 17500 images for training and the rest for validation and save model in folder `models/`.  
Deafult feature extractor is VGG19_bn pretrained on ImageNet. 1 epochs takes 6-7 mins for 1080ti. 

    bash ./yolo_train.sh
#### Data preprocessing
You can change line 39-42 in `yolo_split.py` to split data.
#### Predict by own model
Change model path in `yolo_test.py` (line 38).

## Retinanet case

### Predict by pretrained model
This shell script will load pretraind model `retinanet.pth` and generate `submission.csv` for kaggle.  
Beacause I detect image one by one and use cpu to run nms, so it takes probably 15 min to run this script.

    bash ./retinanet_test.sh
 | kaggle public score | kaggle private score |
 | :--: | :--: |
 | 0.24567 | 0.20000 |

### Train your own model
You can simply run this script to train model, this use first 20000 cases for training and the rest for validation and save model in folder `models/`.  

Same as test, detect image one by one and use cpu nms in validation part.  

Deafult feature extractor is ResNet50 pretrained on ImageNet. 1 epoch take 30-35 mins for 1080ti.

    bash ./retinanet_train.sh
#### Data preprocessing
You can change line 41-45 in `preprocessing.py` to split data.
#### Predict by own model 
Change model path in `retinanet_test.py` (line 30).  

I think the score_threshold(deafult = 0.5) for each retinanet model is different, maybe you should find a suitable value.


## Reference 
1.You Only Look Once: Unified, Real-Time Object Detection [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640.pdf)  
2.Focal Loss for Dense Object Detection [https://arxiv.org/pdf/1708.02002.pdf](https://arxiv.org/pdf/1708.02002.pdf)  
3.Yolov1 code [https://github.com/xiongzihua/pytorch-YOLO-v1](https://github.com/xiongzihua/pytorch-YOLO-v1)  
4.Retinanet code [https://github.com/yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)  
5.NMS cpu version [https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185](https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185) 
 
