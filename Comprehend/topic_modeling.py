import boto3
import json
import time
import pandas as pd

comprehend = boto3.client(service_name='', 
                                      region_name=''
                                      aws_access_key_id='',
                                      aws_secret_access_key=''
                                      )
             
data_access_role_arn = ""
number_of_topics = 15
file_name = ''

input_data_config = {
                    'S3Uri': file_name,
                    'InputFormat': 'ONE_DOC_PER_LINE'
                    }

output_data_config = OutputDataConfig={
                                        'S3Uri': '.../output'
                                      }

start_topics_detection_job_result = comprehend.start_topics_detection_job(
                                                                            NumberOfTopics=number_of_topics,
                                                                            InputDataConfig=input_data_config,
                                                                            OutputDataConfig=output_data_config,
                                                                            DataAccessRoleArn=data_access_role_arn
                                                                          )

job_id = start_topics_detection_job_result["JobId"]
print('job_id: ' + job_id)

while True:
        describe_topics_detection_job_result = comprehend.describe_topics_detection_job(
                                                                                        JobId=job_id
                                                                                        )
        if describe_topics_detection_job_result['TopicsDetectionJobProperties']['JobStatus'] == "COMPLETED":
            print("Completed")
            break
        elif describe_topics_detection_job_result['TopicsDetectionJobProperties']['JobStatus'] == "FAILED":
            print("FAILED")
            break
        elif describe_topics_detection_job_result['TopicsDetectionJobProperties']['JobStatus'] == "STOP_REQUESTED":
            print("STOP_REQUESTED")
            break
        elif describe_topics_detection_job_result['TopicsDetectionJobProperties']['JobStatus'] == "STOPPED":
            print("STOPPED")
            break
        elif describe_topics_detection_job_result['TopicsDetectionJobProperties']['JobStatus'] == "SUBMITTED":
            print("SUBMITTED")
        else:
            print("IN PROGRESS, status would be refreshed every 10 seconds.")
        time.sleep(10)

list_topics_detection_jobs_result = comprehend.list_topics_detection_jobs()

s3_client = boto3.client('s3')
bucket = ""
key = ".../output/output.tar.gz"
input_tar_file = s3_client.get_object(Bucket = bucket, Key = key)
input_tar_content = input_tar_file['Body'].read()
uncompressed_key=".../output/output.json"
import tarfile
from io import BytesIO
with tarfile.open(fileobj = BytesIO(input_tar_content)) as tar:
    for tar_resource in tar:
        if (tar_resource.isfile()):
            inner_file_bytes = tar.extractfile(tar_resource).read()
            s3_client.upload_fileobj(BytesIO(inner_file_bytes), Bucket =bucket,Key=uncompressed_key)
            
txt_file = s3_client.get_object(Bucket = bucket, Key = uncompressed_key)
txt_content = txt_file['Body'].read().decode('utf-8')
xx = [d.strip() for d in txt_content.splitlines()]
tpc = []
trm = []
wt = []
line = []
ln = 0
for i in xx:
    line.append(ln)
    tpc.append(i.split(",")[0])
    trm.append(i.split(",")[1])
    wt.append(i.split(",")[2])
    ln = ln + 1

df = pd.DataFrame({'Topic': tpc, 'Term': trm, 'Weight': wt})
df = df.iloc[1: , :]
