#pip install -U sentence-transformers
#pip install pyvi

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize


# Preprocess_text function
def preprocess_text(text):

    processed_text = tokenize(text)

    return processed_text


# Load model and embedding text funtion
def embedding_text(text):


    PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

    inputs = PhobertTokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    return embeddings.numpy()[0].tolist()
            #torch.Tensor [1, 768] =>numpy.ndarray (1, 768)=> (768,) => list


if __name__ == '__main__':
    text='Thủ tục hành chính lĩnh vực Thi đua, khen thưởng có số thứ tự 01, 02 điểm A1 mục A danh mục 1 ban hành kèm theo Quyết định số 786/QĐ-BVHTTDL ngày 31 tháng 3 năm 2023 của Bộ trưởng Bộ Văn hóa, Thể thao và Du lịch về việc công bố thủ tục hành chính nội bộ giữa các cơ quan, đơn vị trực thuộc Bộ và trong nội bộ cơ quan, đơn vị trực thuộc Bộ thuộc phạm vi chức năng quản lý của Bộ Văn hóa, Thể thao và Du lịch hết hiệu lực thi hành kể từ ngày Quyết định này có hiệu lực thi hành. '
    processed_text=preprocess_text(text)
    embedded_text=embedding_text(text)

