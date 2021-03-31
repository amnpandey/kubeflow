from typing import NamedTuple

import kfp
import kfp.components as components
import kfp.dsl as dsl

from kfp.components import InputPath, OutputPath

NAMESPACE = "kubeflow"


def download_dataset(data_dir: OutputPath(str)):
    """Download the Fraud Detection data set to the KFP volume to share it among all steps"""
    
    from minio import Minio
    import tarfile
    
    url="minio-acme-iaf.apps.sun.cp.fyre.ibm.com"
    key='minio'
    secret='minio123'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    client = Minio(url, key, secret, secure=False)

    client.fget_object('iaf-ai', 'datasets/fraud-detection/datasets.tar.gz', data_dir+'/datasets.tar.gz')
    
   
    tar = tarfile.open(data_dir+'/datasets.tar.gz')
    tar.extractall(path=data_dir)
    tar.close()

def train_model(data_dir: InputPath(str), model_dir: OutputPath(str)):
    """Trains a CNN for 50 epochs using a pre-downloaded dataset.
    Once trained, the model is persisted to `model_dir`."""
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv1D, MaxPool1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model
    
    import pandas as pd
    import numpy as np
    from numpy import asarray
    from numpy import argmax
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print(tf.__version__)
     
    data = pd.read_csv(data_dir+"/datasets/dataset.csv")
    
    print(data.head())
    print(data.shape)
    
    non_fraud = data[data['Class']==0]
    fraud = data[data['Class']==1]
    non_fraud = non_fraud.sample(fraud.shape[0])
    data = fraud.append(non_fraud, ignore_index=True)
    
    print("Class data value count after balancing the dataset")
    print(data['Class'].value_counts())
    
    X = data.drop('Class', axis = 1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    
    print('Training set size: ', len(X_train))
    print(X_train.head(), y_train.head())

    print('Validation set size: ', len(X_test))
    print(X_test.head(), y_test.head())
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # CNN model need 3d array, so need to reshape it
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print("After reshapping the train and test datasets to 3D array")
    print(X_train.shape, X_test.shape)
    
    print("Building the model")
    epochs = 50
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
    
    print("Model Summary: ")
    model.summary()
    
    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.3f' % acc)
    
    print("Saving the model")
    model.save(data_dir+"/"+"model.h5") 
    
    print("Model save successfully.")

def evaluate_model(
    data_dir: InputPath(str), model_dir: InputPath(str), metrics_path: OutputPath(str)) -> NamedTuple("EvaluationOutput", [("mlpipeline_metrics", "Metrics")]):
    """Loads a saved model from file and uses a pre-downloaded dataset for evaluation.
    Model metrics are persisted to `/mlpipeline-metrics.json` for Kubeflow Pipelines
    metadata."""
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv1D, MaxPool1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model
    
    import json
    import pandas as pd
    import numpy as np
    from numpy import asarray
    from numpy import argmax
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    
    #load the data
    data = pd.read_csv(data_dir+"/datasets/dataset.csv")
    
    print(data.head())
    print(data.shape)
    
    non_fraud = data[data['Class']==0]
    fraud = data[data['Class']==1]
    non_fraud = non_fraud.sample(fraud.shape[0])
    data = fraud.append(non_fraud, ignore_index=True)
    
    print("Class data value count after balancing the dataset")
    print(data['Class'].value_counts())
    
    X = data.drop('Class', axis = 1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # load the model from file
    model = load_model(data_dir+"/"+"model.h5")
    #evaluate
    (loss, accuracy) = model.evaluate(X_test)
    print("**")
    print(accuracy)
    metrics = {
        "metrics": [
            {"name": "loss", "numberValue": str(loss), "format": "PERCENTAGE"},
            {"name": "accuracy", "numberValue": str(accuracy), "format": "PERCENTAGE"},
        ]
    }
    
    with open(metrics_path+"/mlpipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

    out_tuple = namedtuple("EvaluationOutput", ["mlpipeline_metrics"])

    return out_tuple(json.dumps(metrics))

def export_model(
    model_dir: InputPath(str),
    metrics: InputPath(str),
    export_bucket: str,
    model_name: str,
    model_version: int,
):
    import os
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-acme-iaf.apps.sun.cp.fyre.ibm.com",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
    )

    # Create export bucket if it does not yet exist
    response = s3.list_buckets()
    export_bucket_exists = False

    for bucket in response["Buckets"]:
        if bucket["Name"] == export_bucket:
            export_bucket_exists = True

    if not export_bucket_exists:
        s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

    # Save model files to S3
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, model_dir)

            s3.upload_file(
                local_path,
                export_bucket,
                f"models/{model_name}/{model_version}/{s3_path}",
                ExtraArgs={"ACL": "public-read"},
            )

    response = s3.list_objects(Bucket=export_bucket)
    print(f"All objects in {export_bucket}:")
    for file in response["Contents"]:
        print("{}/{}".format(export_bucket, file["Key"]))


def train_and_eval(
    workspace: str,
    data_dir: str,
    model_dir: str,
    export_bucket: str,
    model_name: str,
    model_version: int,
):
    # create persistent volume
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="fraud-detection-pvc",
        storage_class='csi-cephfs',
        modes=dsl.VOLUME_MODE_RWM,
        size="10Gi"
    )
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mesosphere/kubeflow:1.0.1-0.5.0-tensorflow-2.2.0"

    downloadOp = components.func_to_container_op(
        download_dataset, base_image=BASE_IMAGE
    )().add_pvolumes({workspace: vop.volume})

    trainOp = components.func_to_container_op(train_model, base_image=BASE_IMAGE)(
        downloadOp.output
    )

    evaluateOp = components.func_to_container_op(evaluate_model, base_image=BASE_IMAGE)(
        downloadOp.output, trainOp.output
    )

    exportOp = components.func_to_container_op(export_model, base_image=BASE_IMAGE)(
        trainOp.output, evaluateOp.output, export_bucket, model_name, model_version
    )
    
    trainOp.after(downloadOp)
    exportOp.after(trainOp)

@dsl.pipeline(
    name="Fraud detection Pipeline",
    description="A sample pipeline to demonstrate multi-step model training, evaluation, export",
)
def fraud_detection_pipeline(
    workspace: str = "/train",
    model_dir: str = "/train/model",
    data_dir: str = "/train/data",
    export_bucket: str = "fraud_detection",
    model_name: str = "fraud_detection",
    model_version: int = 1,
):
    train_and_eval(
        workspace=workspace,
        data_dir=data_dir,
        model_dir=model_dir,
        export_bucket=export_bucket,
        model_name=model_name,
        model_version=model_version
    )
    
pipeline_func = fraud_detection_pipeline
run_name = pipeline_func.__name__ + " run"
experiment_name = "Fraud detection Pipeline"

arguments = {
    "model_dir": "/train/model",
    "data_dir": "/train/data",
    "export_bucket": "fraud_detection",
    "model_name": "fraud_detection",
    "model_version": "1"
}

client = kfp.Client("http://mlpipeline-kubeflow.apps.sun.cp.fyre.ibm.com")
run_result = client.create_run_from_pipeline_func(
    pipeline_func,
    experiment_name=experiment_name,
    run_name=run_name,
    arguments=arguments
)
