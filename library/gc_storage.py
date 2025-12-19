"""
author: Michael Munz

This module contains functions to interact with Google Cloud Storage,
such as uploading or downloading files.
"""

import os
import io

from pathlib import Path
from joblib import dump, load
from google.cloud import storage
from google.api_core import retry

def init_bucket(bucket, json_key_path):
    """initializes Google Cloud storage client
    Required to access files stored in GC

    Args:
        bucket ([type]): name of bucket to access
        json_key_path ([type]): service key JSON file for authentication

    Returns:
        bucket: storage client
    """
    # set up ENV var to JSON key file path 
    # ex: '../../auth/fiery-glass-478009-t8-18a81c8cbe63.json'
    os.environ[ 'GOOGLE_APPLICATION_CREDENTIALS' ] = json_key_path

    # init client
    client = storage.Client()

    # use client to access bucekt
    # ex: 'sep25-bds-road-accidents'
    bucket = client.get_bucket( bucket )
    
    # display
    print( f"Initialized {bucket.name}" )
    
    return bucket


def list_bucket(bucket, remote_folder):
    """list all files in Google Cloud storage folder
    example: 
    remote_folder='2_preprocessing'
    listing all files from 'data/processed/2_preprocessing/'

    Args:
        bucket (object): bucket name
        folder (string): folder to list all joblibs
    """
    remote_prefix = Path('data/processed')
    
    # folder path
    remote_folder_path = remote_prefix / remote_folder
    
    blobs = bucket.list_blobs( prefix=str(remote_folder_path) )

    blob_count = 0
    blob_names = []
    
    # print all files
    for blob in blobs:
        blob_count += 1
        blob_names.append( blob.name )
    
    print( f"Number of blobs: [{blob_count}]" )
    for blob_name in blob_names:
        print( blob_name )


def download(bucket, remote_path):
    """load file from Google Cloud storage
    example:
    remote_path='2_preprocessing/1.1-locations.joblib'
    Downloading file from "data/processed/2_preprocessing/1.1-locations.joblib"

    Args:
        bucket (object): [description]
        remote_path (string): [description]

    Returns:
        [type]: [description]
    """
    remote_prefix = Path('data/processed')
    remote_blob_path = remote_prefix / remote_path
    
    blob = bucket.blob( str(remote_blob_path) )
    
    # navigate to local /data folder
    local_prefix = Path('../..')
    local_path = local_prefix / remote_blob_path
    
    # download to data/processed/<folder>/<file>
    blob.download_to_filename( str(local_path) )
    
    # load local file from data/processed/<folder>/<file>
    local_file = load( str(local_path) )
    
    print( f"Downloaded {str(remote_blob_path)} to\n {str(local_path)}" )
    
    return local_file


def upload(bucket, obj, local_folder, file_name):
    """Save file (dataframe, ML model) to Google Cloud storage
    First it stores file in local folder.
    Then it uploads it to GC storage
    example:
    local_folder='2_preprocessing',
    file_name='1.1-locations.joblib'
    Uploading file to '/data/processed/2_preprocessing/1.1-locations.joblib'
    Args:
        bucket (string)
        obj (pd.DataFrame, ML model)
        local_folder (string): choose any folder name in data/processed/
        file_name (string): the name of the file
    """
    local_path_prefix = Path('../../data/processed')
    
    # 1 local path to file
    local_file_path = local_path_prefix / local_folder / file_name
    
    # 2 store locally
    dump( obj,
          filename=str(local_file_path) )
    
    # 3 remote blob path
    # remove ../../
    remote_blob_path = Path( local_file_path ).relative_to( Path('../../') )
    
    # 4 upload to Google Cloud Storage
    # convert Path -> String
    blob = bucket.blob( str(remote_blob_path) )
    # blob.upload_from_filename( str(local_file_path) )
    with io.open(str(local_file_path), 'rb') as local_file:
        blob.upload_from_file(
            local_file,
            size=-1,
            timeout=120,
            retry=retry.Retry(deadline=600)
        )

    # 4 display
    print( f"Uploaded {str(local_file_path)} to\n gs://{bucket.name}/{str(remote_blob_path)}" )

