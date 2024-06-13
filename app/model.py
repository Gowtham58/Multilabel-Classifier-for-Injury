from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Labels
causes = ['burn or scald - heat or cold exposures - contact with',
 'caught in, under or between',
 'cut, puncture, scrape injured by',
 'fall, slip or trip injury',
 'fall, slip, or trip injury',
 'includes freezing',
 'misc',
 'motor vehicle',
 'rubbed or abraded by',
 'strain or injury by',
 'striking against or stepping on',
 'struck or injured by']
bodyparts = ['head',
 'lower extremities',
 'misc',
 'multiple body parts',
 'neck',
 'trunk',
 'upper extremities']


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#Loading the saved models for the classification task
cause_model = AutoModelForSequenceClassification.from_pretrained("cause_model").to(device)
bodypart_model = AutoModelForSequenceClassification.from_pretrained("bodypart_model").to(device)

def predict(input):
    tk = tokenizer([input])
    tk = {k: torch.Tensor(v).type(torch.int) for k, v in tk.items()}
    tk = {k: v.to(device) for k, v in tk.items()}
    with torch.no_grad():
        outcs = cause_model(**tk)
        outbp = bodypart_model(**tk)
    logitscs = outcs.logits
    predictionscs = torch.argmax(logitscs, dim=-1)  
    logitsbp = outbp.logits
    predictionsbp = torch.argmax(logitsbp, dim=-1)  
    return causes[predictionscs], bodyparts[predictionsbp]

