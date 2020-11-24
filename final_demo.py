import numpy as np
import cv2
import torch
import torchvision
import tarfile
import torch.nn as nn
import torchvision.transforms as T
import time
import joblib
import urllib
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import speech_recognition as sr
import pyttsx3


from classes import ResNet_expression



face_cascade = cv2.CascadeClassifier('/home/praveen/Desktop/Projects/technocolab_project_2/emotion_recognition/src/haarcascade_frontalface_default.xml')
Labels = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Sad',
    5:'Surprise',
    6:'Neutral'
}
stats = ([0.5],[0.5])
tsfm = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(), 
    T.Normalize(*stats,inplace=True)
])

device = torch.device('cuda')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    #xb = img.unsqueeze(0)
    yb = model(img)
    _, preds  = torch.max(yb, dim=1)
    return Labels[preds[0].item()]


input_size = 48*48
output_size = 7

model2 = to_device(ResNet_expression(1, output_size), device)
model2.load_state_dict(torch.load('/home/praveen/Desktop/Projects/technocolab_project_2/emotion_recognition/models/resnet-facial.pth'))






capture_duration = 1
def read_emotion():
    emotion_array = []
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while ( int(time.time() - start_time) < capture_duration ):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            gray = gray.astype(np.uint8)
            gray = cv2.resize(roi_gray, (48, 48))
            transformed_image = tsfm(gray)
            transformed_image = transformed_image.to(device)
            #print(transformed_image.unsqueeze(0))
            #print(' The predicted emotion is:', predict_image(transformed_image.unsqueeze(0), model2))
            emotion_prediction = predict_image(transformed_image.unsqueeze(0), model2)
            if emotion_prediction == "Happy":
                emotion_array.append(3) 
            elif emotion_prediction == "Neutral":
                emotion_array.append(6)
            else:
                emotion_array.append(4)
       
    high_freq_emotion = max(set(emotion_array), key = emotion_array.count) 
    #high_freq_emotion_numpy = np.array(emotion_array).astype(float)
    #high_freq_emotion = np.bincount(high_freq_emotion_numpy).argmax()
    if high_freq_emotion == 3:
        print("You look happy, I'll do my best to make you even more happier")
    elif high_freq_emotion == 6:
        print("You are neither happy/sad so i am ging to make you happy")
    else:
        print("You look sad, i am going to make your search easy and it will make you happy.")
    
        

    cap.release()
    cv2.destroyAllWindows()
    

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:

        print('Listening.....')
        r.pause_threshold = 1 
        audio = r.listen(source)
    try:
        print('Recognizing.....')
        query = r.recognize_google(audio, language="en-US")
        #print(query)
    except Exception as e:
        print(e)
        speak('Say that again Please')
        return 'None'
    return query
    


def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()



def scrapping(question):
    query = question
    query = urllib.parse.quote_plus(query) # Format into URL encoding
    number_result = 1

    ua = UserAgent()
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    response = requests.get(google_url, {"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

    links = []
    titles = []
    descriptions = []
    for r in result_div:
    # Checks if each element is present, else, raise exception
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
        
        # Check to make sure everything is present before appending
            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
    # Next loop if one element is not present
        except:
            continue

    for s in descriptions:
        return s




count_vec = joblib.load("/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/vectorizer_3.pkl")
model = joblib.load("/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/log_reg_count_vec_3.pkl")

count = 0

print("Type 1 for Text I/P and O/P")
print("Type 2 for Audio I/P and O/P")
n = int(input())

while 1:
    
    if n==1:
        sentence = input()
        q_sentence = sentence
    else:
        sentence = take_command()
        q_sentence = sentence
	
    
    if count==0 or count==5:
        read_emotion()
        count=0
    count+=1
    if sentence == "bot close":
        break
    sentence = count_vec.transform([sentence])
    sentence = sentence.toarray()
    
    prediction = model.predict(sentence)
    if prediction == 1:
        print("Hi i am TechBot, I help to optimize your tech search")
    elif prediction == 2:
        reply = scrapping(q_sentence)
        if n==1:
        	print(reply)
        else:
        	#speak(reply)
        	print(reply)

        	
        
    else:
        print("I feel happy that i assisted you, waiting for more questions :))")


