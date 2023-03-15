import speech_recognition
import pyttsx3
from gtts import gTTS
import os
import playsound

def recognizer():
    
    recognize = speech_recognition.Recognizer()
    while True:
        try:
            with speech_recognition.Microphone() as mic:
                recognize.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognize.listen(mic)
                flac_data = audio.get_flac_data()
                with open("my_test_data/audio.flac", "wb") as f:
                    f.write(flac_data)
                text = recognize.recognize_google(audio, language="tr-TR")
                #text = recognize.recognize_google(audio)
                text = text.lower()
                with open('my_test_data/text.txt', 'w') as f:
                    f.write(text)
                break

        except speech_recognition.UnknownValueError():
            recognize = speech_recognition.Recognizer()
            continue
        
def recognizer_eng():
    
    recognize = speech_recognition.Recognizer()
    while True:
        try:
            with speech_recognition.Microphone() as mic:
                recognize.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognize.listen(mic)
                flac_data = audio.get_flac_data()
                with open("my_test_data/sentiment.flac", "wb") as f:
                    f.write(flac_data)
                text = recognize.recognize_google(audio)
                text = text.lower()
                with open('my_test_data/sentiment.txt', 'w') as f:
                    f.write(text)
                break

        except speech_recognition.UnknownValueError():
            recognize = speech_recognition.Recognizer()
            continue


def voiceover(string):
    speech = gTTS(text = string, lang = "tr", slow = False)
    audio_file = "tts_audio.mp3"
    speech.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)
















