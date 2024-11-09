# BR-Pose

BR-Pose: Human Pose Estimation with Bi-Level Routing Attention Mechanism
Installation
Install pytorch == 1.8.0.

Install dependencies:

pip install -r requirements.txt
Install COCOAPI if you want to test in COCO: Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.

Init output(training model output directory) and log(tensorboard log directory) directory:

mkdir output 
mkdir log
Your directory tree should look like this:

1. ${POSE_ROOT}
```
    ├── experiments
    ├── lib
    ├── models
    ├── tools 
    ├── README.md
    └── requirements.txt
    Data preparation
    For MPII data, please download from MPII Human Pose Dataset. The original annotation files are in matlab format. We have converted them into json format, you also need to download them from OneDrive or GoogleDrive. Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
For COCO data, please download from COCO download, 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from OneDrive or GoogleDrive. Download and extract them under {POSE_ROOT}/data, and make them look like this:

${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 