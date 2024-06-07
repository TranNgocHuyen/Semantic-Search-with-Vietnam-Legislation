#pip install -U sentence-transformers
#pip install pyvi

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
import pyodbc
from preprocessing import xu_ly_text
# Preprocess_text function
def preprocess_text(text):
    _, text = xu_ly_text(text) # chọn xử lý theo điều, đoạn
    processed_text = tokenize(text)
    return processed_text


# Load model and embedding text funtion
def embedding_text(text):
    text=preprocess_text(text)
    PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

    inputs = PhobertTokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    return embeddings.numpy()[0].tolist()
            #torch.Tensor [1, 768] =>numpy.ndarray (1, 768)=> (768,) => list

