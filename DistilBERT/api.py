import re
import requests
from typing import Dict
from starlette.requests import Request
from starlette.routing import request_response
from unidecode import unidecode
from string import punctuation

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier import BERTClassifier, get_bert

app = FastAPI()

class ClassificationRequest(BaseModel):
	text: str
	identificador: str
	datetime: str

class ClassificationResponse(BaseModel):
	probabilities: Dict[str, float]
	sentiment: str
	confidence: float

def preProText(text):
	text = text.lower()
	text = re.sub('@[^\s]+', '', text)
	text = unidecode(text)
	text = re.sub('<[^<]+?>','', text)
	text = ''.join(c for c in text if not c.isdigit())
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
	text = ''.join(c for c in text if c not in punctuation)
	return text

def verTermos(text):
	termos = ["suicida", "suicidio", "me matar", "meu bilhete suicida",
	"minha carta suicida", "acabar com a minha vida", "nunca acordar",
	"nao consigo continuar", "nao vale a pena viver", "pronto para pula",
	"pronto pra pular", "dormir pra sempre", "dormir para sempre", 
	"quero morrer", "estar morto", "melhor sem mim", "vou me matar", 
	"melhor morto", "plano de suicidio", "cansado de viver", 
	"morrer sozinho", "não quero estar aqui", "morrer", "matar", 
	"morto", "vida", "descansar em paz"]

	for termo in termos:
		if termo.lower() in text:
			return True

	return False

@app.post("/classifica", response_model = ClassificationResponse)
def classifica(rqt: ClassificationRequest, model: BERTClassifier = Depends(get_bert)):
	texto = preProText(rqt.text)
	identificador = rqt.identificador
	datetime = rqt.datetime

	# POST SERVIDOR
	url = 'https://boamente.minhadespesa.com.br/api/predicoes/store'
	token= 'wocUKkW9GNLxetcJLfirFdPsTfiBkv4eH4pfG7k2Lu8'

	if (verTermos(texto) == True):
		sentiment, confidence, probabilities = model.predict(texto)
			
		probabilidade = round(float(confidence),5)
		possibilidade = int(sentiment)

		payload={
        	'token': token,
        	'identificador': identificador,
        	'probabilidade': probabilidade, 
        	'possibilidade': possibilidade,
        	'data_criacao': datetime
        	}
			
		resposta = requests.post(url, data=payload)
		
		print(texto, probabilidade, possibilidade, resposta.status_code,': ' , resposta.text)

		return ClassificationResponse(
			sentiment = sentiment, 
			confidence = confidence, 
			probabilities = probabilities
		)
	else:
		probabilidade = 0.0
		possibilidade = 0

		payload={
        	'token': token,
        	'identificador': identificador,
        	'probabilidade': probabilidade, 
        	'possibilidade': possibilidade,
        	'data_criacao': datetime
        	}
			
		resposta = requests.post(url, data=payload)
		
		print(texto, probabilidade, possibilidade, resposta.status_code,': ' , resposta.text)