#!/usr/bin/env python
# coding: utf-8

# In[5]:


#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import os
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3
#get_ipython().magic(u'matplotlib inline')

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction






# In[6]:

# Define a transform to pre-process the testing images
transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader
data_loader = get_loader(transform=transform_test,    
                         mode='test')


# In[7]:


# Load the most recent checkpoint
checkpoint = torch.load(os.path.join('/home/osboxes/image_captioning/example', 'train-model-112000.pkl'))

# Specify values for embed_size and hidden_size - we use the same values as in training step
embed_size = 256
hidden_size = 512

# Get the vocabulary and its size
vocab = data_loader.dataset.vocab
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Move models to GPU if CUDA is available.
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()


# In[8]:


x=get_prediction(data_loader, encoder, decoder, vocab)


# In[9]:


print(x)


# In[ ]:
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate',rate-50)
engine.say(x)
engine.runAndWait()




# In[ ]:




