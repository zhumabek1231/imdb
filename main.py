import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchtext  import datasets, data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from fastapi import FastAPI
import uvicorn
classes=['positive','negative']


class CheckIMDB(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 64)
    self.lstm = nn.LSTM(64, 128, batch_first=True)
    self.fc = nn.Linear(128, 2)

  def forward(self,x):

    x = self.embedding(x)

    _, (hidden_state, _) = self.lstm(x)

    x = hidden_state.squeeze(0)
    x = self.fc(x)

    return x



imdb_app = FastAPI()
vocab = torch.load('vocab_imdb.pth', map_location='cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CheckIMDB(len(vocab)).to(device)
model.load_state_dict(torch.load('imdb3.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = get_tokenizer('basic_english')
def change_words(text):
  return [vocab[i] for i in tokenizer(text)]


class TextShema(BaseModel):
    word:str


@imdb_app.post('/predict')
async def change_text(text:TextShema):
    num_text = torch.tensor(change_words(text.word)).unsqueeze(0).to(device)


    with torch.no_grad():
        pred = model(num_text)
        result = torch.argmax(pred,dim=-1).item()
        return {'class':classes[result]}


if __name__=='__main__':
    uvicorn.run(imdb_app,host='127.0.0.1',port=8000)