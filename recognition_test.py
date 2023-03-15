import torch
import numpy as np
from fuzzywuzzy import fuzz
from IPython.display import clear_output
import random
import shutil
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import myconfig
import neural_net
import feature_extraction
import speechTotext_textTospeech
from sentiment_analysis import predict_tweet_recognition


def string_distance_metrics(string_1, string_2):
    """Measuring similarity between main text and spoken text"""
    distance = fuzz.ratio(string_1, string_2)
    return distance


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_new_data(max_score):
    """Adding the data to the data if there is no data in the encoder"""
    text = "Kim olduğunuzu bulamadık. Eğer isminizi yazarsanız sizi verilere ekleyebiliriz."
    speechTotext_textTospeech.voiceover(text)
    name = input("İsminiz: ")
    src_path = myconfig.MY_TEST_DATA_DIR + "audio.flac"
    dst_path = myconfig.MY_TRAIN_DATA_DIR + "audio.flac"
    shutil.copy2(src_path, dst_path)
    os.chdir(myconfig.MY_TRAIN_DATA_DIR)
    os.rename('audio.flac','{}.flac'.format(name))
        
          

def encoder_dataset(path,encoder):
    """Converting the sounds in the train data into encoder"""
    data_encode = {}
    datalist = os.listdir(path)
    for data in datalist:
        name = data.split(".")[0]
        data = path + data
        features = feature_extraction.extract_features(data)
        sliding_windows = feature_extraction.extract_sliding_windows(features)
        if not sliding_windows:
            return None
        batch_input = torch.from_numpy(
            np.stack(sliding_windows)).float().to(myconfig.DEVICE)
        batch_output = encoder(batch_input)
        aggregated_output = torch.mean(batch_output, dim=0, keepdim=False).cpu()
        aggregated_output = aggregated_output.data.numpy()
        data_encode[name] = aggregated_output
    return data_encode


def scores(path, names, encoders):
    """Calculating the cosine similarity scores of the voice in the test data 
    between the voices in the train data"""
    score = []
    
    for test_data in os.listdir(path):
        if "audio" in test_data:
            data = path + test_data
            features = feature_extraction.extract_features(data)
            sliding_windows = feature_extraction.extract_sliding_windows(features)
            if not sliding_windows:
                return None
            batch_input = torch.from_numpy(
                np.stack(sliding_windows)).float().to(myconfig.DEVICE)
            batch_output = encoder(batch_input)
            aggregated_output = torch.mean(batch_output, dim=0, keepdim=False).cpu()
            aggregated_output = aggregated_output.data.numpy()
            for encode in encoders:
                score.append(cosine_similarity(encode, aggregated_output))
        
    return score

        
def all_results(names, cosine_similarity_score,str1, str2):
    """Show and catch all result"""
    print("Similarity rates:")
    for name, score in zip(names, cosine_similarity_score):
        print(f"{name}: {score}")
    print("--------------------")
    print("Maximum similarity:")
    cosine_similarity_score = np.array(cosine_similarity_score)
    cosine_similarity_score = cosine_similarity_score[0:len(names)]
    max_ind = np.argmax(cosine_similarity_score)
    print(f"{names[max_ind]}: {cosine_similarity_score[max_ind]}")
    with open(str2) as f:
        for line in f:
            str2 = line
    distance = string_distance_metrics(str1, str2)
    return names[max_ind], cosine_similarity_score[max_ind], distance

def results_voiceover(name, score, distance):
    """Soice the results"""
    text_for_name = "Merhaba {}".format(name)
    text = ""
    if score < 0.50:
        add_new_data(score)
    else:
        if distance >= 80:
            text = "İstenilen metin ile söylediğiniz metin eşleşiyor. Sistemimizi kullanabilirsiniz. Sentiment analiz yapmak istiyorsanız 1 istemiyorsanız 0 tuşuna basınız."
            #print(f"spoken text is %{distance} compatible with the main text")
            speechTotext_textTospeech.voiceover(text_for_name)
            speechTotext_textTospeech.voiceover(text)
            choice = input("Your choice: ")
            clear_output(wait=True)
            return choice
        else:
            text = "İstenilen metin ile söylediğiniz metin eşleşmiyor. Sistemimizi kullanamazsınız." 
            speechTotext_textTospeech.voiceover(text_for_name)
            speechTotext_textTospeech.voiceover(text)
    

def sentiment():
    text = "ingilizce bir cümle söyleyin."
    speechTotext_textTospeech.voiceover(text)
    speechTotext_textTospeech.recognizer_eng()
    with open(myconfig.SENTIMENT_TEXT_DIR) as f:
        for line in f:
            text = line
    predict_tweet_recognition(text)
    
# Test data to calculate cosine similarity
test_data_path = myconfig.MY_TEST_DATA_DIR
# Text path to text-prompt system
text_data_path = myconfig.TEXT_DIR

# Sentence the speaker should say and the part where the speaker's voice is recorded
sentence = random.choice(myconfig.SENTENCE_LIST)
print("Alttaki cümleyi söylemeniz gerekiyor:", "\n", sentence)
test_audio = speechTotext_textTospeech.recognizer()

# Train data encoder
encoder = neural_net.get_speaker_encoder(
    myconfig.SAVED_MODEL_PATH)
encoder_dict = encoder_dataset(myconfig.MY_TRAIN_DATA_DIR, encoder)

# Data's names and their encoder
names = list(encoder_dict.keys())
encoders = list(encoder_dict.values())

cosine_similarity_score = scores(test_data_path, names, encoders)
name, score, distance = all_results(names, cosine_similarity_score,sentence.lower(), text_data_path)
choice = results_voiceover(name, score, distance)
if choice == "1":
    sentiment()







     

