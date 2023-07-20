from __future__ import print_function
import time
import boto3

transcribe = boto3.client('',
                          region_name = '' 
)

file_uri = '.../sample.mp3'
vocab_name = 'vocabulary_table'
vocab_filter_name = 'vocabulary_filter_table'

def create_vocab(file_uri):
    vocab_name = file_uri.rpartition('/')[-1].rpartition('.')[-3]
    transcribe.create_vocabulary(
                                    LanguageCode = 'en-US',
                                    VocabularyName = vocab_name,
                                    VocabularyFileUri = file_uri
                                )
    while True:
        status = transcribe.get_vocabulary(VocabularyName = vocab_name)
        if status['VocabularyState'] in ['READY', 'FAILED']:
            return status['VocabularyState']
        print("Not ready yet...")
        time.sleep(5)
    print(status)
    
def create_vocab_filter(file_uri):
    vocab_name = file_uri.rpartition('/')[-1].rpartition('.')[-3]
    transcribe.create_vocabulary_filter(
                                    LanguageCode = 'en-US',
                                    VocabularyFilterName = vocab_name,
                                    VocabularyFilterFileUri = file_uri
                                )
    print('Transcribe Vocabulary Filter Created Successfully')
    return None
    
def delete_vocab(vocab_name):
    transcribe.delete_vocabulary(
                                    VocabularyName=vocab_name
                                )
    print('Transcribe Vocabulary deleted successfully')
    return None

def delete_vocab_filter(vocab_name):
    transcribe.delete_vocabulary_filter(
                                            VocabularyFilterName=vocab_name
                                        )
    print('Transcribe Vocabulary Filter deleted successfully')
    return None
    
def get_vocab(vocab_name):
    try:
        response = transcribe.get_vocabulary(
                                                VocabularyName=vocab_name
                                            )
        print('Transcribe Vocabulary Fetched successfully')
        return response
    except Exception as e:
        return e
    
def get_vocab_filter(vocab_name):
    try:
        response = transcribe.get_vocabulary_filter(
                                                        VocabularyFilterName=vocab_name
                                                    )
        print('Transcribe Vocabulary Filter Fetched successfully')
        return response
    except Exception as e:
        return e
    
def update_vocab(vocab_name, file_uri):
    try:
        transcribe.update_vocabulary(
                                        VocabularyName=vocab_name,
                                        LanguageCode='en-US',
                                        VocabularyFileUri=file_uri
                                    )
        print('Transcribe Vocabulary Update Successful')
        return True
    except Exception as e:
        return e
    
def update_vocab_filter(vocab_name, file_uri):
    try:
        transcribe.update_vocabulary_filter(
                                    VocabularyFilterName=vocab_name,
                                    LanguageCode='en-US',
                                    VocabularyFilterFileUri=file_uri
                                )
        print('Transcribe Vocabulary Update Successful')
        return True
    except Exception as e:
        return e

def create_job(file_uri, vocab_name, vocab_filter_name):
    job_name = "job-" + file_uri.rpartition('/')[-1]
    try:
        transcribe.start_transcription_job(
                                            TranscriptionJobName=job_name,
                                            Media={'MediaFileUri': file_uri},
                                            MediaFormat='mp3',
                                            # MediaFormat={'mp3'|'mp4'|'wav'|'flac'|'ogg'|'amr'|'webm'},
                                            LanguageCode='en-US',
                                            OutputBucketName = "",
                                            OutputKey = "output/",
                                            ContentRedaction = { 
                                                                'RedactionType':'PII', 
                                                                'RedactionOutput':'redacted_and_unredacted'
                                                                # 'PiiEntityTypes': ['ALL']
                                                                },
                                            Settings = {
                                                        'VocabularyName':vocab_name,
                                                        'VocabularyFilterName': vocab_filter_name,
                                                        'VocabularyFilterMethod': 'mask'
                                                        # 'VocabularyFilterMethod': 'remove'|'mask'|'tag'
                                                        }
                                            )
        return True
    except Exception as e:
        return e

def get_transcription_text(file_uri):
    job_name = "job-" + file_uri.rpartition('/')[-1]
    while True:
        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = job['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            print(f"Job {job_name} completed")
            print()
            # with urllib.request.urlopen(job['TranscriptionJob']['Transcript']['TranscriptFileUri']) as r:
            #     data = json.loads(r.read())
            # return data['results']['transcripts'][0]['transcript']
            return None
        elif status == 'FAILED':
            print(f"Job {job_name} failed")
            print()
            return None
        else:
            print(f"Status of job {job_name}: {status}. Status would be refreshed every 5 seconds.")
        time.sleep(5)
    
def delete_job(file_uri):
    job_name = "job-" + file_uri.rpartition('/')[-1]
    transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    print('Job deleted successfully')
    return None
    
    
create_job(file_uri, vocab_name, vocab_filter_name)
get_transcription_text(file_uri)
print("The transcribed text for" + file_uri.rpartition('/')[-1] + " file is stored in S3 successfully")
print()
delete_job(file_uri)
