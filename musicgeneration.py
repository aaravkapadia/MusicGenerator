import comet_ml
COMET_API_KEY = "APIKEY"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from IPython.display import clear_output
from music21 import converter
from music21 import midi
import re

TRAIN = False
songs = "PATHTOABCFILE"
text = open(songs).read()
songs_joined = text
vocab = sorted(set(songs_joined))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
def vectorize_string(string):
    return np.array([char2idx[c] for c in string], dtype = np.int64)
vectorized_songs = vectorize_string(songs_joined)
def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch = [vectorized_songs[i:i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1:i + seq_length+1] for i in idx]
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)
    return x_batch, y_batch
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size, device):
        return(torch.zeros(1, batch_size, self.hidden_size).to(device),
            torch.zeros(1, batch_size, self.hidden_size).to(device))
    
    def forward(self, x, state = None, return_state= False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0),x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)
    
vocab_size = len(vocab)
device = torch.device("cpu")


cross_entropy = nn.CrossEntropyLoss()
def compute_loss(labels, logits):
    batched_labels = labels.view(-1)
    batched_logits = logits.view(-1, logits.size(-1))
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

params = dict(
  epochs = 1000, 
  batch_size = 8, 
  seq_length = 100, 
  learning_rate = 1e-3, 
  embedding_dim = 256,
  hidden_size = 1024, 
)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

def create_experiment():
  if 'experiment' in locals():
    experiment.end()
  experiment = comet_ml.Experiment(
                  api_key=COMET_API_KEY,
                  project_name="music-generation")
  for param, value in params.items():
    experiment.log_parameter(param, value)
  experiment.flush()
  return experiment

model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"])
model.to(device)
model.load_state_dict(torch.load(checkpoint_prefix, map_location=device))
model.eval()
optimizer = optim.Adam(model.parameters(), lr= params["learning_rate"])

def train_step(x, y):
  model.train()
  optimizer.zero_grad()
  y_hat = model(x)
  loss = compute_loss(y, y_hat)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
  optimizer.step()
  return loss
if TRAIN:
    history = []
    experiment = create_experiment()

    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()

    for iter in tqdm(range(params["epochs"])):
        x_batch, y_batch = get_batch(
            vectorized_songs,
            params["seq_length"],
            params["batch_size"]
        )
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train_step(x_batch, y_batch)
        experiment.log_metric("loss", loss.item(), step=iter)
        history.append(loss.item())
        if iter % 100 == 0:
            torch.save(model.state_dict(), checkpoint_prefix)

    torch.save(model.state_dict(), checkpoint_prefix)
    experiment.flush()

def generate_text(model, start_string, generation_length=1000):
    model.eval()
    input_idx = vectorize_string(start_string)
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)
    state = model.init_hidden(input_idx.size(0), device)
    text_generated = []
    tqdm._instances.clear()
    for i in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True) # TODO
        predictions = predictions[:, -1, :]
        probs = torch.softmax(predictions, dim=-1)
        input_idx = torch.multinomial(probs, num_samples=1)
        text_generated.append(idx2char[input_idx.item()])
    return start_string + ''.join(text_generated)


generated_text = generate_text(model, start_string="X:1\n", generation_length=1000)
def sanitize_abc(abc_text):
    abc_text = re.sub(
        r'[^A-Ga-gzZ|:\[\]/0-9_=^,\'>\nKMLTX ]',
        '',
        abc_text
    )
    abc_text = re.sub(r'(\^|_|=)(?![A-Ga-g])', '', abc_text)
    parts = abc_text.split("\nX:")
    abc_text = parts[0] if len(parts) == 1 else "X:" + parts[1]
    if not abc_text.startswith("X:"):
        abc_text = "X:1\n" + abc_text

    if "T:" not in abc_text:
        abc_text = abc_text.replace("X:1\n", "X:1\nT:Generated Tune\n")

    if "M:" not in abc_text:
        abc_text = abc_text.replace("T:", "M:4/4\nT:")

    if "L:" not in abc_text:
        abc_text = abc_text.replace("M:", "L:1/8\nM:")

    if "K:" not in abc_text:
        abc_text += "\nK:C\n"

    return abc_text.strip()
clean_abc = sanitize_abc(generated_text)
abc_path = os.path.expanduser("~/generated_song.abc")

with open(abc_path, "w") as f:
    f.write(clean_abc)

print(f"Saved to {abc_path}")
score = converter.parse(abc_path)

midi_path = os.path.expanduser("~/generated_song.mid")
score.write("midi", midi_path)

print(f"Saved MIDI to {midi_path}")




