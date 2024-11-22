import torch
import datasets
import csv
from data import create_dataset
import pandas as pd
from datasets import Dataset
from options import Options
import os
import requests
import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import sys
from evaluate import load

sys.path.append('.')

splits = {'validation': 'simplification/validation-00000-of-00001.parquet', 'test': 'simplification/test-00000-of-00001.parquet'}
train_data = pd.read_parquet("hf://datasets/facebook/asset/" + splits["validation"])
val_data = pd.read_parquet("hf://datasets/facebook/asset/" + splits["test"])[:352]

train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)

opt = Options()
train_data = create_dataset(opt, train_data)
val_data = create_dataset(opt, val_data)


def plot_plt_losses(self, epoch, counter_ratio, losses):
    if not hasattr(self, 'plot_data'):
        self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
    self.plot_data['X'].append(epoch + counter_ratio)
    self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
    X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), axis=1)
    Y = np.array(self.plot_data['Y'])
    legend = self.plot_data['legend']

    plt.figure(figsize=(10, 6))
    for i in range(len(legend)):
        plt.plot(X[:, 0], Y[:, i], label=legend[i])  # Plotting the first column of X against each column of Y

    plt.title(self.name + ' loss over time')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    # Save the plot to a PNG file
    plt.savefig(f'{self.name} loss.png')
    plt.close()  # Close the figure to free memory

def train(net, train_data, val_data, epochs=10, n_seqs=10, n_steps=50, lr=1e-5, clip=5, val_frac=0.1, cuda=False, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        n_seqs: Number of mini-sequences per mini-batch, aka batch size
        n_steps: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        cuda: Train with CUDA on a GPU
        print_every: Number of steps for printing training and validation loss

    '''

    bertscore = load("bertscore")
    bleu = load("bleu")
    rouge = load('rouge')
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    # val_idx = int(len(data)*(1-val_frac))
    # data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        net.cuda()

    counter = 0
    # n_chars = len(net.chars)

    old_time = time.time()
    losses = []

    best_loss = float('inf')

    for e in tqdm(range(epochs)):
        h = net.init_hidden(n_seqs)
        for data in train_data:
            src = data["original"]
            tgt = list(data["simplifications"][0])

            # print(src)
            # print(tgt)

            tokenized = tokenizer(src+tgt, truncation=True, padding=True, max_length=1024, return_tensors='pt')

            if counter%2 == 0:
                cur_ids =  tokenized["input_ids"][:len(src)]
                cur_mask = tokenized["attention_mask"][:len(src)]
            else:
                cur_ids =  tokenized["input_ids"][len(src):]
                cur_mask = tokenized["attention_mask"][len(src):]

            x = dict()
            x["input_ids"] = cur_ids[:,:-1]
            x["attention_mask"] = cur_mask[:,:-1]

            y = dict()
            y["input_ids"] = cur_ids[:,1:]
            y["attention_mask"] =cur_mask[:,1:]

            # if time.time() - old_time > 60:
            #     old_time = time.time()
            #     requests.request("POST",
            #                      "https://nebula.udacity.com/api/v1/remote/keep-alive",
            #                      headers={'Authorization': "STAR " + response.text})

            counter += 1

            # One-hot encode our data and make them Torch tensors
            inputs = x["input_ids"]
            targets = y["input_ids"]
            targets[y["attention_mask"] == 0] = -100
            
            # print("targets:", targets.shape)
            # print("inputs:", inputs.shape)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()

            if cuda:
                output = net.forward(inputs, x["attention_mask"].cuda(), hidden_in=h[0], mem_in=h[1])
            else:
                output = net.forward(inputs, x["attention_mask"], hidden_in=h[0], mem_in=h[1])

            h = (output.lstm_hidden_state, output.lstm_memory)

            output = output.logits
            # print("output", output.shape)
            # print("targets", targets.flatten().shape)

            loss = criterion(output, targets.flatten())
            losses.append(loss.item())
            print(f"loss: {loss}", end='\r')

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            optimizer.step()

            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(n_seqs)
                val_losses = []
                for j, data in tqdm(enumerate(val_data)):
                    # One-hot encode our data and make them Torch tensors
                    src = data["original"]
                    tgt = list(data["simplifications"][0])

                    # print(src)
                    # print(tgt)

                    tokenized = tokenizer(src+tgt, truncation=True, padding=True, max_length=1024, return_tensors='pt')

                    if j%2 == 0:
                        cur_ids =  tokenized["input_ids"][:len(src)]
                        cur_mask = tokenized["attention_mask"][:len(src)]
                    else:
                        cur_ids =  tokenized["input_ids"][len(src):]
                        cur_mask = tokenized["attention_mask"][len(src):]

                    x = dict()
                    x["input_ids"] = cur_ids[:,:-1]
                    x["attention_mask"] = cur_mask[:,:-1]

                    y = dict()
                    y["input_ids"] = cur_ids[:,1:]
                    y["attention_mask"] =cur_mask[:,1:]

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs = x["input_ids"]
                    targets = y["input_ids"]
                    targets[y["attention_mask"] == opt.pad_token_id] = -100
                    # print("targets:", targets.shape)

                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output = net.forward(inputs, x["attention_mask"].to(inputs.device), hidden_in=val_h[0], mem_in=val_h[1])
                    
                    logits = output.logits
                    # print("logits", logits.shape)
                    # print("targets", targets.flatten().shape)

                    val_loss = criterion(logits, targets.flatten())

                    val_losses.append(val_loss.item())

                gen_text = [tokenizer.decode(token) for token in output.gen_tokens]
                target_text = [tokenizer.decode(target) for target in targets]

                # Calculate BERT score
                bert_score = bertscore.compute(predictions=gen_text, references=target_text, lang="en")
                bert_f1 = np.mean(bert_score['f1'])
                bert_p = np.mean(bert_score['precision'])
                bert_r = np.mean(bert_score['recall'])

                # Calculate BLEU
                bleu_sc = bleu.compute(predictions=gen_text, references=target_text)
                bleu_sc = bleu_sc['bleu']

                # Calculate Rouge
                rouge_sc = rouge.compute(predictions=gen_text, references=target_text)
                rouges1 = np.mean(rouge_sc['rouge1'])
                rouges2 = np.mean(rouge_sc['rouge2'])
                rougesL = np.mean(rouge_sc['rougeL'])

                print("----- Sample Generation")
                print("predicted:", gen_text)
                print("target:", target_text)

                # save checkpoint
                if val_loss < best_loss:
                    print("save checkpoint..")
                    model_name = f'{directory}g_pretrained.net'

                    checkpoint = {'n_hidden': net.hidden_size,
                                'n_layers': net.num_layers,
                                'state_dict': net.state_dict(),}

                    with open(model_name, 'wb') as f:
                        torch.save(checkpoint, f)

                print("save the loss log")
                # append loss data to csv file
                csv_file = f'{directory}pt_BERT_loss.csv'

                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # check if the file is empty to write header
                    if file.tell() == 0:
                        writer.writerow(["train_loss", "val_loss", "bert_f1", "bert_p", "bert_r", "bleu", "rouge1", "rouge2", "rougeL"])

                    writer.writerow([np.mean(losses), np.mean(val_losses), bert_f1, bert_p, bert_r, bleu_sc, rouges1, rouges2, rougesL])

                losses = []

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


if opt.emb == "bart":
    print("bart tokenizer")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
else:
    print("bert tokenizer")
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

from model import LSTMG, LSTMG_latest


directory = f'./checkpoints/{opt.name}/'
if not os.path.exists(directory):
    os.makedirs(directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = LSTMG_latest(opt, device=device)
# Load model

# with open('./weight2/g_pretrained.net', 'rb') as f:
#     checkpoint = torch.load(f)

# print("Load checkpoints..")
# net.load_state_dict(checkpoint['state_dict'])
# print(net)

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
    cuda=True
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')
    cuda=False

n_seqs, n_steps = opt.batch_size, 60

# you may change cuda to True if you plan on using a GPU!
# also, if you do, please INCREASE the epochs to 25

train(net, train_data, val_data, epochs=10, n_seqs=n_seqs, n_steps=n_steps, lr=1e-3, cuda=cuda, print_every=5)

# Close the training log file.
f.close()
