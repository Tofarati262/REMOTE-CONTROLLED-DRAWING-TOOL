import speech_recognition as sr
from enum import Enum

class Language(Enum):
    ENGLISH = "en-US"
    CHINESE = "zh-TW"
    FRENCH = "fr-FR"

class SpeechToText():
    def print_mic_device_index():
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("{1}, device_index={0}".format(index,name))
    
    def speech_to_text(device_index, language=Language.ENGLISH):
        r = sr.Recognizer()
        text=""
        with sr.Microphone(device_index=device_index) as source:
            print("Start Talking:")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language=language.value)
                print("You said: {}".format(text))
            except:
                print("please try again.")
        return text

def check_mic_device_index():
    SpeechToText.print_mic_device_index()

def run_speech_to_text_english(device_index):
    SpeechToText.speech_to_text(device_index)

def run_speech_to_text_french(device_index):
    SpeechToText.speech_to_text(device_index, Language.FRENCH)

def constant_check(device_index):
    while True:
        return run_speech_to_text_english(device_index)
       


if __name__ == "__main__":
    #print_mic_device_index()
    
    constant_check(3)

            