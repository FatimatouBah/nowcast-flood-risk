# -*- coding=utf-8 -*-
# ============================
# IMPORT LIBRAIRIES
# ============================

# gestion dossiers et fichiers
import os
from pathlib import Path
import sys
# librairies pour se connecter à AWS S3 et manipuler les données dans le bucket
import boto3
# Import datetime locally to validate date values after regex extraction.
from datetime import datetime
# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv


# Absolute path of this script's folder: .../Dev/python_apps/data_collection
CURRENT_DIR = Path(__file__).resolve().parent
# Add folder to import search path once
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import convert_date_to_datetime, parse_s3_uri


# *==========================================================================================
# *1. CONFIGURATION : aws s3
# *==========================================================================================

# ************* ATTENTION : INTERDIRE D'AFFICHER/IMPRIMER LES CREDENTIELS d'AWS S3*************
# 1. get IAM AWS credentials from env variables if they exist, otherwise use the default ones (user_access_key and user_secret_key)
load_dotenv(dotenv_path=Path.cwd().resolve() / ".env") # load env variables from .env file in the current directory
# load_dotenv() # load env variables from .env file in the current directory
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_PROFILE_NAME = os.getenv("AWS_S3_PROFILE_NAME", "default")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
EXISTING_BUCKET_MAIN = os.getenv("EXISTING_BUCKET_MAIN")
S3_RESOURCE_ROOT_FOLDER = os.getenv("S3_RESOURCE_ROOT_FOLDER")
# ************* ATTENTION : INTERDIRE D'AFFICHER/IMPRIMER LES CREDENTIELS *************

# Specific folder in your s3 bucket where to store the data inside the root folder, you can change it as you want
# if the root folder <Final_Project_Forecasting> does not exist in your s3 bucket, it will be created automatically when you upload the first file to it, 
# same for the subfolder <Dataset/hubeau_api> 
S3_RESOURCE_DATA_FOLDER = f"{S3_RESOURCE_ROOT_FOLDER}/Dataset/hubeau_api"

# choose the storage folder in your s3 bucket where to upload the json and csv files
S3_STORAGE_ALL = f"{S3_RESOURCE_DATA_FOLDER}/ALL"
S3_STORAGE_VARS = f"{S3_RESOURCE_DATA_FOLDER}/VARS"


# 2. Create an instance of `boto3.Session` that connects with your aws account.
# Create a low-level service client by name using the default session.
# A session stores configuration state and allows you to create service clients and resources
aws_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
    profile_name=AWS_S3_PROFILE_NAME
)

# 3. Create a variable called `s3` that connects your session to the s3 ressource.
# Create a resource service client by name: create an S3 resource using the aws session
# The resource provides an object-oriented API as well as low-level access to AWS services. See the documentation for details on the service's resource APIs.
# service_name : string. The name of a service, e.g. 's3' or 'ec2'. You can get a list of available services via get_available_resources.
s3_resource = aws_session.resource("s3")

print("S3 resource service created successfully:", s3_resource)
# Tester la connexion en listant les buckets de votre compte S3 resource
print("\nBuckets in your S3 resource:", s3_resource.buckets.all())
# List all buckets in your S3 ressources service
print("\nBuckets in your S3 resource:")
for bucket in s3_resource.buckets.all():
    print(f" - ressource - Bucket: {bucket.name}")

# 4. Create a variable will connect to an existing bucket in your s3 or to a bucket you are creating now.
# Create a bucket resource object for the existing bucket using the s3 resource
S3_BUCKET_RESOURCE = s3_resource.Bucket(EXISTING_BUCKET_MAIN)

print(f"\nCalling the bucket object: {S3_BUCKET_RESOURCE}")
print(f"Using Bucket Name: {S3_BUCKET_RESOURCE.name}")
print(f"Bucket Creation Date: {S3_BUCKET_RESOURCE.creation_date}")
# print(f"Bucket ACL: {S3_BUCKET_RESOURCE.Acl()}")

# 5. list all objects in the bucket to check if the connection is successful and to see the existing files in the bucket
# Final_Project_Forecasting/Dataset/hubeau_api
# print(f"\nObjects in the bucket {S3_BUCKET_RESOURCE.name}:")
# for obj in S3_BUCKET_RESOURCE.objects.all():
#     if obj.key.startswith(S3_RESOURCE_DATA_FOLDER):
#         print(f" - object - Key: {obj.key}, Size: {obj.size} bytes, Last Modified: {obj.last_modified}")
#         print(f"   URI: s3://{S3_BUCKET_RESOURCE.name}/{obj.key}")
#         s3_uri = f"s3://{S3_BUCKET_RESOURCE.name}/{obj.key}"
#         print(f"URI: {s3_uri}")
#         print(parse_s3_uri(s3_uri))


## Recupération des URI des fichiers stockés dans AWS S3 pour une utilisation ultérieure dans le projet
s3_ALL_DATABASE_dict = {"site_name":[], "code_site":[], "code_station":[], "s3_csv_uri":[], "s3_json_uri":[], "start_date":[], "end_date":[]}
s3_VAR_DATABASE_dict = {"site_name":[], "code_site":[], "code_station":[], "Measure":[], "s3_csv_uri":[], "s3_json_uri":[], "start_date":[], "end_date":[]}

"""
main path format types to respect for the files stored in the S3 bucket, which are defined in the upload_to_s3.py script when we upload the files to S3:
- hubeau_obs_elab_{site_name}_{code_site}_{code_station}_ALL_{start_date}_{end_date}
- hubeau_obs_elab_{site_name}_{code_site}_{code_station}_{grandeur_hydro_elab}_{start_date}_{end_date}

- Grandeurs hydrométriques (grandeur_hydro_elab) élaborées disponibles : 
    - débits moyens journaliers (QmnJ), 
    - débits moyens mensuels (QmM), 
    - Hauteur instantanée maximale mensuelle (HIXM), 
    - Hauteur instantanée maximale journalière (HIXnJ), 
    - Débit instantané minimal mensuel (QINM), 
    - Débit instantané minimal journalier (QINnJ), 
    - Débit instantané maximal mensuel (QixM), 
    - Débit instantané maximal journalier (QIXnJ)
"""

print(f"\nObjects in the bucket {S3_BUCKET_RESOURCE.name}:")
for obj in S3_BUCKET_RESOURCE.objects.all():
    # Final_Project_Forecasting/Dataset/hubeau_api
    if obj.key.startswith(f"{S3_RESOURCE_DATA_FOLDER}/ALL/cleaned"):
        s3_uri = f"s3://{S3_BUCKET_RESOURCE.name}/{obj.key}"
        site_name, code_site, code_station, measure, start_date, end_date = parse_s3_uri(s3_uri)
        print("ALL:", site_name, code_site, code_station, measure, start_date, end_date)
        s3_ALL_DATABASE_dict["s3_csv_uri"].append(s3_uri)
        s3_ALL_DATABASE_dict["s3_json_uri"].append(s3_uri)
        s3_ALL_DATABASE_dict["site_name"].append(site_name)
        s3_ALL_DATABASE_dict["code_site"].append(code_site)
        s3_ALL_DATABASE_dict["code_station"].append(code_station)
        s3_ALL_DATABASE_dict["start_date"].append(start_date)
        s3_ALL_DATABASE_dict["end_date"].append(end_date)
    elif obj.key.startswith(f"{S3_RESOURCE_DATA_FOLDER}/VARS/cleaned"):
        s3_uri = f"s3://{S3_BUCKET_RESOURCE.name}/{obj.key}"
        site_name, code_site, code_station, measure, start_date, end_date = parse_s3_uri(s3_uri)
        print("VARS:", site_name, code_site, code_station, measure, start_date, end_date)
        s3_VAR_DATABASE_dict["s3_csv_uri"].append(s3_uri)
        s3_VAR_DATABASE_dict["s3_json_uri"].append(s3_uri)
        s3_VAR_DATABASE_dict["site_name"].append(site_name)
        s3_VAR_DATABASE_dict["code_site"].append(code_site)
        s3_VAR_DATABASE_dict["code_station"].append(code_station)
        s3_VAR_DATABASE_dict["Measure"].append(measure)
        s3_VAR_DATABASE_dict["start_date"].append(start_date)
        s3_VAR_DATABASE_dict["end_date"].append(end_date)