"""
    Please do not alter any key names in config_dict,
    otherwise it may cause serious problem that lead
    to the training failure.


    After altering train_proportion,
    please run :

        python3 faceProcessor.py --split

    in order to update dataset.


    Please change the location of the train and test
    dataset path and the output path in faceProcessor
    simultaneously to prevent unexpected error.
"""


from torchvision import transforms
from modeling.xception import xception
from modeling.networks import *
from matplotlib.patches import Patch
from logging import INFO, DEBUG

config_dict = {
    "dataset": {
        "name": "combine_dataset",
        "train": "../../Bosheng/ff_c40/combine/train",#../data/combine/train/   ../../Bosheng/ff/combine/ cross1/uadfv/
        "test": "../../Bosheng/ff_c40/combine/test",#../data/combine/test/
        "train_proportion": 0.85,
        "target": {
            0: "real",
            1: "fake"
        },
        "transform": transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    },
    "BusterNet": {
        "batch_size": 2,
        "learning_rate": 5e-4,  # start at 5e-4 then 5e-7
        "n_epochs": 10,
        "momentum": 0,
        "train_amount": 330000,
        "test_amount": 59000,
        "embeddingNet": xception(pretrained=False),#  EmbeddingNet4(),
        "classification": {
            "n_epochs": 200,
            "learning_rate": 3e-3,
            "network": ClassificationNet1()
        }
    },
    "Memorizer": {
        "base": "memory_ff_all",
        "load_model_epoch": None,
        "logLevel": INFO,
        "logFile": "buster.log",
        "logFormat": "time: %(asctime)s  %(name)s: %(levelname)s\n%(message)s\n",
        "checkpoint": "checkpoint.pth",
        "classifierModel": "classifierModel.pth",
        "embeddingData": {"train": {"data": "train_embedding_data_",
                                    "label": "train_embedding_label_"},
                          "test": {"data": "test_embedding_data_",
                                   "label": "test_embedding_label_"},
                          "tail": ".npy"},
        "dataset": {"train": {"data": "train_data_",
                              "label": "train_label_"},
                    "test": {"data": "test_data_",
                             "label": "test_label_"},
                    "tail": ".npy"}
    },
    "faceProcessor": {
        "imageSize": (299, 299),
        "modelPath": "shape_predictor_68_face_landmarks.dat",
        "rawVideo": {
            "fake": ["../FaceForensics/c40/manipulated_sequences/Deepfakes/c40/videos/"],#"../data/CelebDF/Celeb-synthesis/"    ../uadfv_data/videos/fake/  ../FaceForensics/original_sequences/youtube/c23/videos
            "real": ["../FaceForensics/c40/original_sequences/youtube/c40/videos/"]       #../FaceForensics/manipulated_sequences/NeuralTextures/c23/videos/
        },
        "out": {
            "real": "../../Bosheng/facedata/c40/real/",#./faceData/origin/real/
            "fake": "../../Bosheng/facedata/c40/fake/NeuralTextures/",#./faceData/origin/fake/
        },
        "amountEachVideo": 35,
    },
    "visualization": {
        "losses": {
            "color": ["#ebc334", "#3462eb"],
            "legend": [Patch(color="#ebc334", label="Train Avg Losses"),
                       Patch(color="#3462eb", label="Test Avg Losses")]
        },
        "classification": {
            "color": ["#ebc334", "#3462eb"],
            "legend": [Patch(color="#3462eb", label="Loss"),
                       Patch(color="#ebc334", label="Accuracy")]
        },
        "embeddings": {
            "color": ["#ebc334", "#3462eb"],
            "legend": [Patch(color="#ebc334", label="Real"),
                       Patch(color="#3462eb", label="Fake")]
        }
    },
    "compare": {
        "baselinePath": "../test/",
        "preTrainedModel": "../faceforensics++_models_subset/face_detection/xception/all_c23.p",
        # "preTrainedModel": "../faceforensics++_models_subset/full/xception/full_c23.p"
    }
}



# Current training dataset =>
