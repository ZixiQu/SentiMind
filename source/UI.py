if __name__ == "__main__":
   print("The system will start in a few second ...")

import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# for plotting
import matplotlib.pyplot as plt

# for algebric computations
import numpy as np
from pprint import pprint
from tqdm import tqdm

# We may choose a GPU if we have one on our machine
if torch.backends.cuda.is_built():
  # if we have cuda
  # usually on Windows machines with GPU
  device = "cuda"
elif torch.backends.mps.is_built():
  # if we have MPS
  # usually on MAC
  device = "mps"
else:
  # if not we should use our CPU
  device = "cpu"


id_to_class = {
    0: 'Admiration',
    1: 'Amusement',
    2: 'Anger',
    3: 'Annoyance',
    4: 'Approval',
    5: 'Caring',
    6: 'Confusion',
    7: 'Curiosity',
    8: 'Desire',
    9: 'Disappointment',
    10: 'Disapproval',
    11: 'Disgust',
    12: 'Embarrassment',
    13: 'Excitement',
    14: 'Fear',
    15: 'Gratitude',
    16: 'Grief',
    17: 'Joy',
    18: 'Love',
    19: 'Nervousness',
    20: 'Optimism',
    21: 'Pride',
    22: 'Realization',
    23: 'Relief',
    24: 'Remorse',
    25: 'Sadness',
    26: 'Surprise',
    27: 'Neutral'
    }


class CNN_Model(torch.nn.Module):
    def __init__(self, embedding_size):
        """
        k1 k2: conv layer size, conv layer size: k1 x embedding_size
        n1 n2: channel #
        """
        super().__init__()
        k1 = 2
        k2 = 4
        # channel_size1 = 10
        # channel_size2 = 20

        kernal_size1 = (k1, embedding_size)
        kernal_size2 = (k2, embedding_size)

        # self.conv1 = torch.nn.Conv2d(1, channel_size1, kernal_size1, bias=False)
        self.conv1_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=kernal_size1),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(k1, 1)),
        ])
        self.batchnorm1 = nn.BatchNorm2d(256)


        # self.conv2 = torch.nn.Conv2d(1, channel_size2, kernal_size2, bias=False)
        self.conv2_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=kernal_size2),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(k2, 1)),
        ])
        self.batchnorm2 = nn.BatchNorm2d(256)
        # Linear Layers:
        self.line1 = nn.Linear(512, 2048)
        self.line2 = nn.Linear(2048, 1024)
        self.line3 = nn.Linear(1024, 28)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm3 = nn.BatchNorm1d(2048)
        self.batchnorm4 = nn.BatchNorm1d(1024)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedded_text):
        # embedded_text: [2, 25, 768] batch 2, all sentences max length 25, embedding size 768
        sentence_emb = embedded_text.unsqueeze(1)  # for conv2d()

        sentence_len = sentence_emb.size(2)
        x1, x2 = sentence_emb, sentence_emb

        for conv in self.conv1_layers:
            x1 = self.dropout(x1)
            x1 = F.relu(conv(x1))
            x1 = self.batchnorm1(x1)
            

        for conv in self.conv2_layers:
            x2 = self.dropout(x2)
            x2 = F.relu(conv(x2))
            x2 = self.batchnorm2(x2)

        # conv1_out = self.conv1(sentence_emb)
        # conv1_out: [2, 10, 24, 1] batch 2, channel 10, output length 24, dummy 1
        # conv2_out = self.conv2(sentence_emb)
        # conv1_out: [2, 20, 22, 1] batch 2, channel 10, output length 22, dummy 1

        # conv1_out = torch.nn.functional.relu(conv1_out)
        # conv2_out = torch.nn.functional.relu(conv2_out)

        conv1_out = x1.squeeze(3)
        conv2_out = x2.squeeze(3)

        # maxpool
        conv1_out = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)  # [batch_size, n1]
        # conv1_out: [2, 10]
        conv2_out = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)  # [batch_size, n2]

        after_maxpool = torch.cat((conv1_out, conv2_out), 1)
        after_maxpool = self.dropout(after_maxpool)

        line1_out = F.relu(self.line1(after_maxpool))
        line1_out = self.batchnorm3(line1_out)
        line1_out = self.dropout(line1_out)
        
        line2_out = F.relu(self.line2(line1_out))
        line2_out = self.batchnorm4(line2_out)
        line3_out = self.line3(line2_out)
        # return torch.sigmoid(self.output_model(after_maxpool))  # apply sigmoid
        distribution = line3_out
        return self.sigmoid(distribution)
    

def infer(probs, threshold=0.5):
    """ return a list of detected emotions. If none pass threshold return single emotion that has highest value
    probs: return result of CNN
    threshold: default 0.5 (sigmoid activation, 0.5 means positive values before activation)

    return torch.tensor([0 if not included else 1]) of size probs.shape[-1], where probs should be shape [batch_size, class #]
    """
    if probs.ndim == 1:
        # If probs is of shape [class #], add a batch dimension
        probs = probs.unsqueeze(0)  # Shape becomes [1, class #]
    
    # Apply threshold to get binary detections
    detected = (probs >= threshold).int()
    
    # Handle case where no emotions are detected
    none_detected = (detected.sum(dim=1) == 0)
    if none_detected.any():
        # For entries with no detections, set the highest probability to 1
        highest_prob_indices = probs.argmax(dim=1)
        detected[none_detected, :] = 0  # Reset all detections for the affected rows
        detected[none_detected, highest_prob_indices[none_detected]] = 1
    
    return detected

    
# Choose a model
model_name = "roberta-base"  # roberta-base albert-base-v2
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder_model = AutoModel.from_pretrained(model_name).to(device)
embedding_size = encoder_model.config.hidden_size

def context_aware_embed(texts):
    """ 
    using encoder-only transformers, process the input texts to context-aware embeddings for downstream tasks
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    return outputs.last_hidden_state


CNN = CNN_Model(embedding_size).to(device)
with open(f'CNN_8854_3851.pt', "rb") as f:  # CNN_8854_3851.pt CNN_1layer_4038.pt
    CNN.load_state_dict(torch.load(f))

CNN.eval()

count = 1
def process_text(input_text):
    global count
    print(f'{count} message: {input_text}')
    count += 1
    
    probs = CNN(context_aware_embed(input_text))
    emotions = infer(probs).nonzero(as_tuple=True)[1].tolist()
    result = [id_to_class[each] for each in emotions]
    return ", ".join(result)
    

if __name__ == '__main__':
    
    with gr.Blocks() as demo:
        gr.Markdown("# SentiMind!")
        gr.Markdown("What emotions reside in your sentences?")
        
        with gr.Column():
            input_box = gr.Textbox(lines=5, label="Input")
            output_box = gr.Textbox(label="Output")
            submit_button = gr.Button("Submit")
    
        # Define the interaction
        submit_button.click(fn=process_text, inputs=input_box, outputs=output_box)

    # Launch the interface
    demo.launch(share=True)