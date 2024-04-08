# Machine Learning Engineer Test: Computer Vision and Object Detection

## Objective
This test aims to assess your skills in computer vision and object detection, with a specific focus on detecting room walls and identifying rooms in architectural blueprints or pre-construction plans.

This test evaluates your practical skills in applying advanced computer vision techniques to a specialized domain and your ability to integrate machine learning models into a simple API server for real-world applications.

Choose one of the visual tasks, one of the text extraction tasks, and the API Server task. We encourage you to submit your tests even if you can’t complete all tasks.

Good luck!


## Full test description
[Senior Machine Learning Engineer.pdf](https://github.com/TrueBuiltSoftware/ml-eng-test/files/14545316/Senior.Machine.Learning.Engineer.1.pdf)

## PS
Share your project with the following GitHub users:
- vhaine-tb
- omasri-tb
- alexwine36

## Cloning repository
This repository uses 2 submodules.
1. https://github.com/LeoProPlus/CubiCasa5k
    - Fork of the official CubiCasa5k dataset repo. It is used for training. We utilize a few functions in order to read CubiCasa5k dataset.
2. https://github.com/LeoProPlus/ml-eng-test-lfs-storage
    - Forked repositories cannot use git lfs, so we added another submodule to work around this problem. We are using git lfs to store trained models.

 The best way to clone everything at once is to run the following command (it may take a while because it also downloads lfs files):

```
git clone --recurse-submodules https://github.com/LeoProPlus/ml-eng-test
```

## Installation

To start application run the following command from project root directory:
```
docker compose -f src/docker-compose.dev.yml up --build
```

This command runs two applications:
1. Webserver for serving API available at http://127.0.0.1:5000/
2. JupyterLab available at http://127.0.0.1:8888/lab

## API Testing
For testing purposes you can use `Swagger` available at [http:/localhost:5000](http://127.0.0.1:5000/) or you can use cURL.

### Example cURL
```
curl -X POST 'http://127.0.0.1:5000/predict?type=walls' -F 'image=@src\ml\walls_detection\data\examples\F1_original.png'
curl -X POST 'http://127.0.0.1:5000/predict?type=tables' -F 'image=@src\ml\table_extraction\data\input\image_1.jpg'
```
