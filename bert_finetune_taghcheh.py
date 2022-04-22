# -*- coding: utf-8 -*-
"""BERT-Finetune-Taghcheh"""

# install transformers
!pip install -q transformers

# import libraries
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from imblearn.under_sampling import RandomUnderSampler

from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup

from tqdm import tqdm, trange
from collections import Counter

import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

# set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

df = pd.read_csv("cleaned_taghche.csv")
df

df.dtypes

df.rate.unique()

df["rate"] = df["rate"].astype(int)

df.rate.unique()

# y = LabelBinarizer().fit_transform(df.rate)

"""### Delete rows with rate 5 and keep only 8000 samples"""

df[df.rate == 5].shape[0]
n_5_rates = df[df.rate == 5].shape[0]

number_to_keep = 10000

n_remove = n_5_rates - number_to_keep

to_remove = np.random.choice(df[df['rate']==5].index,size=n_remove,replace=False)
df_balanced = df.drop(to_remove)

df_balanced[df_balanced.rate == 5]

df_balanced

"""### Make 2 class rate classification"""

df_balanced['rate_binary'] = (df_balanced['rate'] >= 4).astype(int)

df_balanced

# set data and target
comments = df_balanced.cleaned_comment.values
rate = df_balanced.rate_binary.values

type(comments)

# add BERT tokens to the sentence
comments = ["[CLS]" + comment + "[SEP]" for comment in comments.astype(str)]

comments[0:10]

# tokenize sentences
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case = True)
tokenized_comment = [tokenizer.tokenize(text) for text in comments]

# convert tokens to ids
input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenized_comment]

# pad the sequence
MAX_LEN = 128
padded_ids = pad_sequences(input_ids, truncating= 'post', padding='post', maxlen= MAX_LEN, dtype = 'long')

# create attention mask for sentences
att_mask = []
for seq in padded_ids:
    masked_data = [float(tok>0) for tok in seq]
    att_mask.append(masked_data)

# define test and train data
train_inp, valid_inp, train_label, valid_label = train_test_split(padded_ids, rate, random_state= 2018, test_size=0.2)
train_mask, valid_mask, _, _ = train_test_split(att_mask, padded_ids, random_state= 2018, test_size=0.2 )

# convert data to torch tensor format
train_inp = torch.tensor(train_inp)
valid_inp = torch.tensor(valid_inp)
train_label = torch.tensor(train_label)
valid_label = torch.tensor(valid_label)

train_mask = torch.tensor(train_mask)
valid_mask = torch.tensor(valid_mask)

train_label.shape

valid_label.shape

valid_label[0]

# loading mini-batch data
BATCH_SIZE = 32

train_data = TensorDataset(train_inp, train_mask, train_label)
train_sample = RandomSampler(train_data)
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, sampler= train_sample)

valid_data = TensorDataset(valid_inp, valid_mask, valid_label)
valid_sample = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, sampler= valid_sample)

# call BERT model for Classification
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels = 2)
model.cuda()

# weight decay 
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
# Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
{'params': [p for n, p in param_optimizer if not any(nd in n for nd
in no_decay)],
'weight_decay_rate': 0.1},
# Filter for parameters which *do* include those.
{'params': [p for n, p in param_optimizer if any(nd in n for nd in
no_decay)],
'weight_decay_rate': 0.0}
]

#The Hyperparameters for the Training Loop

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

optimizer = AdamW(optimizer_grouped_parameters,
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )
# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives us the number of batches.
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis =1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat)/len(label_flat)

torch.cuda.empty_cache()

#The Training Loop
t = []
# Store our loss and accuracy for plotting
train_loss_set = []
# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc = 'EPOCH'):
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    # Tracking variables
    tr_loss = 0
    nb_tr_example, nb_tr_steps = 0 ,0
    # Train the data for one epoch
    for step, batch in enumerate(train_loader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
        loss = output["loss"]
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_example += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set

    model.eval()
    # Tracking variables 
    eval_loss , eval_accuracy = 0, 0
    nb_eval_example, nb_eval_steps = 0, 0
    # Evaluate data for one epoch
    for batch in valid_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
        # Move logits and labels to CPU
        logits = logits['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate Validation Accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        nb_eval_example += b_input_ids.size(0)

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# plot the training loss 
plt.figure(figsize = (15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()



