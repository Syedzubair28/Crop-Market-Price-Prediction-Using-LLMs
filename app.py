from flask import Flask, render_template, request, flash, redirect
import sqlite3
import pickle
import numpy as np
import random
import requests
import warnings
import google.generativeai as genai
import sqlite3
from googletrans import Translator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import requests
import base64
translator = Translator()

def load_model(model_path):
    """Load the trained model and tokenizer"""
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_cost(tokenizer, model, state, district, year, season, crop):
    """Make a prediction"""
    text = (
        f"State: {state}, District: {district}, "
        f"Year: {year}, Season: {season}, Crop: {crop}"
    )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits.item()

# Load model (update path if needed)
MODEL_PATH = "./fast_crop_model"
tokenizer, model = load_model(MODEL_PATH)

# Load Gemini AI model
genai.configure(api_key='AIzaSyC67rrRtCfo9J3rd5ziUnhuRInlCtL3m24')
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
chat = gemini_model.start_chat(history=[])

warnings.filterwarnings('ignore')

chat_history = []
chat_history_kn = []
chat_history_hi = []
def TTS(text1, lng):
    myobj = gTTS(text=text1, lang=lng, tld='com', slow=False)
    myobj.save("voice.mp3")
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()
    
connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS sessions(name TEXT, password TEXT, timestamp TEXT)"""
cursor.execute(command)

app = Flask(__name__)
chat_history = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('userlog.html')

@app.route('/uhome')
def uhome():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()
        print(name,password)
        if result:
            from datetime import datetime
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            cursor.execute("INSERT INTO sessions VALUES ('"+name+"', '"+password+"', '"+str(date_time)+"')")
            connection.commit()
            return render_template('logged.html')
        else:
            return render_template('logged.html', msg='Sorry , Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        query = "SELECT * FROM user WHERE mobile = '"+mobile+"'"
        cursor.execute(query)

        result = cursor.fetchone()
        if result:
            return render_template('index.html', msg='Phone number already exists')
        else:

            cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
            connection.commit()

            return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route("/kidneyPage")
def kidneyPage():
    return render_template('logged.html')


@app.route("/predictPage", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST': 
        state = request.form['state']
        district = request.form['district']
        year = request.form['year']
        season = request.form['season']  
        crop = request.form['crop']  
        
        # Call your prediction function
        cost = predict_cost(tokenizer, model, state, district, year, season, crop)
        print(f"\n\n\n{cost}\n\n\n")
        dd="cost of the crop is "+ str(cost)

        # Prepare the data to pass to template
        result = {
            'state': state,
            'district': district,
            'year': year,
            'season': season,
            'crop': crop,
            'cost': f"{cost:.2f}"
        }
        gemini_response = chat.send_message("give me only numerical value of price for crop"+ crop +"in state"+state + "of district" + district+"in a season"+season +", don't want any other text,if you do not know give me atleast approximate price between 500 to 3000")
        response_text = gemini_response.text
        print(response_text)
        TTS(response_text,'kn')
        TTS(response_text,'en')
        TTS(response_text,'hi')
        # Render template with only the variables you have
        return render_template('predict.html', result=result,result1=str(response_text))
    
    return render_template('logged.html')
    # return render_template('logged.html')

    return render_template('logged.html')
@app.route('/kannada', methods=['GET', 'POST'])
def kannada():
    if request.method == 'POST':
        user_input = request.form['query']  

        if user_input.lower() in ["hi", "hello", "ನಮಸ್ತೆ", "ನಮಸ್ಕಾರ"]:
            response_text = "ಹಾಯ್, ನಾನು ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
        else:
            from_lang = 'kn'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text   

            gemini_response = chat.send_message(user_input)
            response_text = gemini_response.text
            print(f"Bot: {response_text}")

            from_lang = 'en'
            to_lang = 'kn'
            text_to_translate = translator.translate(response_text, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            TTS(text, 'kn')

            chat_history_kn.append([user_input, text])
            return render_template('kannada.html',chat_history_kn=chat_history_kn)

        TTS(response_text, 'kn')

        chat_history_kn.append([user_input, response_text])
        return render_template('kannada.html', chat_history_kn=chat_history_kn)

    return render_template('kannada.html')

@app.route('/english', methods=['GET', 'POST'])
def english():
    if request.method == 'POST':
        user_input = request.form['query']
        try:
            if any(greeting in user_input.lower() for greeting in ["hi", "hello"]):
                response_text = "Hi, how may I assist you today?"
            else:
                gemini_response = chat.send_message(user_input)
                response_text = gemini_response.text
                print(f"Bot: {response_text}")
            
            TTS(response_text, 'en')
            chat_history.append([user_input, response_text])
            return render_template('english.html', chat_history=chat_history)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('error.html', message=error_message)
    return render_template('english.html')

@app.route('/hindi', methods=['GET', 'POST'])
def hindi():
    if request.method == 'POST':
        user_input = request.form['query']  

        if user_input.lower() in ["hi", "hello", "नमस्ते"]:
            response_text = "नमस्ते, आज मैं आपकी कैसे सहायता कर सकता हूँ?"

            
        else:
            from_lang = 'hi'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text  
            
            gemini_response = chat.send_message(user_input)
            response_text = gemini_response.text
            print(f"Bot: {response_text}")

            from_lang = 'en'
            to_lang = 'hi'
            text_to_translate = translator.translate(response_text, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            TTS(text, 'hi')

            chat_history_hi.append([user_input, text])
            return render_template('hindi.html', chat_history_hi=chat_history_hi)

        TTS(response_text, 'hi')

        chat_history_hi.append([user_input, response_text])
        return render_template('hindi.html', chat_history_hi=chat_history_hi)

    return render_template('hindi.html')
@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
##    if request.method == 'POST':
##        user_input = request.form['query']
##        # Get response from Gemini AI model
##        gemini_response = chat.send_message(user_input)
##
##        data = gemini_response.text
##        print(data)
##        result = []
##        for row in data.split('*'):
##            if row.strip() != '':
##                result.append(row)
##        print(result)
##        # Update chat history with both responses
##        chat_history.append([user_input, result])
##
##        return render_template('chatbot.html', chat_history=chat_history)
    return render_template('userlog.html')

if __name__ == '__main__':
	app.run(debug = True, use_reloader=False)
