import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import itertools
import csv
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartScaledWordEmbedding, BartLearnedPositionalEmbedding
from transformers import AutoConfig, BartTokenizer, BertTokenizer, EncoderDecoderModel, BartConfig
import math
import numpy as np
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass

# Define modeling output class
@dataclass
class Seq2SeqGenerator(ModelOutput):
    """
    Source: transformers/modeling_outputs.py
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        decoder_hidden_state (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits                      : torch.FloatTensor = None
    loss                        : Optional[torch.FloatTensor] = None
    gen_tokens                  : Optional[torch.LongTensor] = None
    decoder_hidden_state        : Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state   : Optional[torch.FloatTensor] = None
    encoder_hidden_states       : Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions          : Optional[Tuple[torch.FloatTensor, ...]] = None
    lstm_hidden_state           : Optional[Tuple[torch.FloatTensor, ...]] = None
    lstm_memory                 : Optional[Tuple[torch.FloatTensor, ...]] = None

class LSTMG(nn.Module):
    def __init__(self, opt, num_layers=1, hidden_size=128, device="cpu", drop_prob=0.25):
        super(LSTMG, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = opt.vocab_size

        shared = BartScaledWordEmbedding(self.vocab_size, opt.d_model, opt.pad_token_id)

        if opt.emb == "bart":
            print("BART Embedding")
            config = BartConfig.from_pretrained("facebook/bart-large-cnn")
            self.embedding = BartEncoder(config, shared)
            self.embedding.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/NTUST/Thesis Topic/LLM/My Text GAN/VBartCycle/Split File/checkpoints/encoder_weights_BART.dict', weights_only=True))
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                print("gpu embedding")
                self.embedding.cuda()
        elif opt.emb == "t5":
            from transformers import T5Model
            model = T5Model.from_pretrained('google/t5-efficient-large')
            self.embedding = model.encoder
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                self.embedding.cuda()
        else:
            print("BERT Embedding")
            model_name = 'patrickvonplaten/bert2bert_cnn_daily_mail'
            model = EncoderDecoderModel.from_pretrained(model_name, output_hidden_states=True) # Load the full model, but only use the encoder part
            self.embedding = model.encoder
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                self.embedding.cuda()

        self.lstm = nn.LSTM(input_size=opt.d_model, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=drop_prob)
        
        self.dropout = nn.Dropout(drop_prob)

        # Map the output to token Probs
        self.lm_head = nn.Linear(opt.d_model, self.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_seq, input_mask, hidden_in=None, mem_in=None):
        # LSTM forward
        batch_size = input_seq.size(0)
        # if hidden_in == None:
        #     # Initialize the memory buffers
        #     hidden_in = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_seq.device)
        #     mem_in = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_seq.device)
        print("batch_size:", batch_size)
        
        # Embedding
        print("input_seq:", input_seq.shape)
        print("input_mask:", input_mask.shape)
        # print("input_seq:", input_seq)
        # print("input_mask:", input_mask)
        # Change token padding from -100 to 1
        input_seq[input_seq == -100] = 1
        print("input_seq", input_seq.device)
        print("input_mask", input_mask.device)
        if len(input_seq.shape) < 2:
            input_seq = input_seq.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
        input_embs = self.embedding(input_seq, input_mask)
        print("input_embs:", input_embs[0].shape)

        # LSTM
        output, (hidden_out, mem_out) = self.lstm(input_embs[0], (hidden_in, mem_in))
        x = output.contiguous()
        print("output:", x.shape)
        print("hidden_out:", hidden_out.shape)
        print("mem_out:", mem_out.shape)

        # pass x through a droupout layer
        x = self.dropout(x)

        x = x.view(-1, self.hidden_size)

        # Projection to probability
        logits = self.lm_head(x)
        print("logits:", logits.shape)

        # Find generated tokens using greedy algorithm
        # max_p_tokens = torch.max(logits, 2)
        # gen_tokens = max_p_tokens.indices
        # gen_tokens = gen_tokens.long()
        # print("gen_tokens:", gen_tokens.shape)

        return logits, (hidden_out, mem_out)

    def predict(self, word, tokenizer, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
        #x = np.array([[self.char2int[char]]]
        # x = F.one_hot(x, self.vocab_size)
        inputs = x["input_ids"][:,1]
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs.unsqueeze(0), x["attention_mask"][:,1].unsqueeze(0), hidden_in=h[0], mem_in=h[1])

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ch = np.arange(self.vocab_size)
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        word = np.random.choice(top_ch, p=p/p.sum())

        return tokenizer.decode(word), h

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        # self.lm_head.bias.data.fill_(0)
        # FC weights as random uniform
        self.lm_head.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.num_layers, n_seqs, self.hidden_size).zero_())


class LSTMG_latest(nn.Module):
    def __init__(self, opt, device=torch.device("cpu"), drop_prob=0.25):
        super(LSTMG_latest, self).__init__()

        self.batch_size = opt.batch_size
        self.num_layers = opt.n_layers_G
        self.hidden_size = opt.hidden_size
        self.vocab_size = opt.vocab_size
        self.max_seq_len = opt.seq_length
        self.gamma = opt.gamma
        self.pad_token_id =  opt.pad_token_id
        self.device = device

        if opt.emb == "bart":
            print("BART Embedding")
            shared = BartScaledWordEmbedding(self.vocab_size, opt.d_model, opt.pad_token_id)
            config = BartConfig.from_pretrained("facebook/bart-large-cnn")
            self.embedding = BartEncoder(config, shared)
            self.embedding.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/NTUST/Thesis Topic/LLM/My Text GAN/VBartCycle/Split File/checkpoints/encoder_weights_BART.dict', weights_only=True))
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                print("gpu embedding")
                self.embedding.cuda()
        elif opt.emb == "t5":
            from transformers import T5Model
            model = T5Model.from_pretrained('google/t5-efficient-large')
            self.embedding = model.encoder
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                self.embedding.cuda()
        else:
            print("BERT Embedding")
            model_name = 'patrickvonplaten/bert2bert_cnn_daily_mail'
            model = EncoderDecoderModel.from_pretrained(model_name, output_hidden_states=True) # Load the full model, but only use the encoder part
            self.embedding = model.encoder
            for param in self.embedding.parameters():
                param.requires_grad = False
                param.to(device)
            if device == "gpu":
                self.embedding.cuda()

        self.E = nn.LSTM(input_size=opt.d_model, hidden_size=int(self.hidden_size//2),
                         num_layers=self.num_layers, batch_first=True, dropout=drop_prob, bidirectional=True)

        # The decoder input will be the probability of each vocab
        self.D = nn.LSTM(input_size=opt.hidden_size, hidden_size=self.hidden_size,
                         num_layers=self.num_layers, batch_first=True, dropout=drop_prob)

        self.dropout = nn.Dropout(drop_prob)

        # Map the output to token Probs
        self.lm_head = nn.Linear(opt.d_model, self.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_seq, input_mask, hidden_in=None, mem_in=None):
        # LSTM forward
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)
        if hidden_in is None:
            self.init_hidden(batch_size)
            hidden_in = self.h[0]
            mem_in = self.h[1]

        input_seq[input_seq == -100] = self.pad_token_id

        if len(input_seq.shape) < 2:
            input_seq = input_seq.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)

        # Embedding
        input_embs = self.embedding(input_seq, input_mask)

        ## LSTM Encoder ##
        output_e, (hidden_e, mem_e) = self.E(input_embs[0], (hidden_in, mem_in))

        ## LSTM Decoder ##
        sents = []
        logits = []
        # concat bidirection info
        output_e = output_e.reshape(batch_size, seq_len, -1)
        hidden_e = hidden_e.reshape(self.num_layers, batch_size, -1)
        mem_e = mem_e.reshape(self.num_layers, batch_size, -1)

        # print("output_e", output_e.shape)
        # print("hidden_e", hidden_e.shape)
        # print("mem_e", mem_e.shape)

        output, (hidden_d, mem_d) = self.D(output_e, (hidden_e, mem_e))
        x = output.contiguous()
        # pass x through a droupout layer
        x = self.dropout(x)
        sents.append(x)
        # Projection to probability
        logits = self.lm_head(x.view(-1, self.hidden_size))
        logits = logits/self.gamma

        logits = logits.view(batch_size,seq_len,self.vocab_size)
        sents = torch.cat(sents).view(batch_size,-1,self.hidden_size)

        # Find generated tokens using greedy algorithm (instead of beam search, for faster training)
        max_p_tokens = torch.max(logits, 2)
        gen_tokens = max_p_tokens.indices
        gen_tokens = gen_tokens.long()
        eot_idx = (gen_tokens == 102).nonzero()
        # if len(eot_idx) > 0:
        #     for i, j in eot_idx:
        #         gen_tokens[i,j+1:] = 0
        print("gen_tokens:", gen_tokens[0])

        # return (logits/self.gamma), (hidden_d, mem_d)
        return Seq2SeqGenerator(
            logits=logits.view(-1, self.vocab_size),
            gen_tokens=gen_tokens,
            decoder_hidden_state=sents,  # dim: bs, seq, hidden
            encoder_last_hidden_state=output_e, # dim: bs, seq, hidden
            lstm_hidden_state=hidden_e.reshape(self.num_layers*2, batch_size, -1),
            lstm_memory=mem_e.reshape(self.num_layers*2, batch_size, -1)
        )

    def generate(self, vt, h):

        ## LSTM Decoder ##
        output, (hidden_d, mem_d) = self.D(vt.view(self.batch_size,1,self.vocab_size), h)
        # print("output shape:", output.shape)
        x = output.contiguous()
        # pass x through a droupout layer
        x = self.dropout(x)
        # print("x shape:", x.shape)

        # Projection to probability
        p = self.lm_head(x.view(-1, self.hidden_size))
        p = F.softmax((p/self.gamma), dim=1)
        # print("p shape:", p.shape)

        return p, (hidden_d, mem_d)

    def predict(self, x, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(self.batch_size)

        # x = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
        #x = np.array([[self.char2int[char]]]
        # x = F.one_hot(x, self.vocab_size)
        inputs = x
        # print("inputs:",inputs.shape)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        
        out, h = self.generate(inputs, h)

        p = out
        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ch = np.arange(self.vocab_size)
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # print("-p:", p)
        # print("top_ch:", top_ch.shape)
        # print("p:", p.shape)

        p = p.detach().numpy().squeeze()
        ps = []
        if len(p.shape) > 1:
            for i,candidate in enumerate(p):
                ps.append(np.random.choice(top_ch[i], p=candidate/candidate.sum()))
        else:
            ps.append(np.random.choice(top_ch, p=p/p.sum()))
        # print("--p:", p)

        return np.array(ps), out, h

    def sample(self, tokenizer, size, prime='The', top_k=None, cuda=False):

        if cuda:
            self.cuda()
        else:
            self.cpu()

        self.eval()

        # LSTM forward
        input= tokenizer(prime, return_tensors="pt", truncation=True, padding=True)
        input_seq = input["input_ids"].to(self.device)
        input_mask = input["attention_mask"].to(self.device)
        # print("seq length:", input_seq.shape[0])
        self.init_hidden(input_seq.shape[0])
        h = self.h

        # Change token padding from -100 to 1
        input_seq[input_seq == -100] = 0

        # Embedding
        input_embs = self.embedding(input_seq, input_mask)

        ## LSTM Encoder ##
        output_e, h = self.E(input_embs[0], (h[0], h[1]))

        # First off, run through the prime characters
        words = [[] for _ in range(self.batch_size)]
        vt = input_embs[0][:,0,:] # get the <sot> of each batch
        logits = self.lm_head(vt.view(-1, self.hidden_size))
        vt = F.softmax((logits/self.gamma), dim=1)
        # print("vt:", vt.shape)
        # print("vt view:", vt.view(1,1,self.vocab_size).shape)
        for i in range(self.max_seq_len):
            ps, vt, h = self.predict(vt, h, cuda=cuda, top_k=top_k)
            print("ps:", ps)
            # print("p:", ps.shape)
            for j, p in enumerate(ps):
                word = tokenizer.decode(p)
                words[j].append(word)
                if p == 102:
                    # add padding
                    for i in range((self.max_seq_len-1)-i):
                        words.append(0)
                    break
            # logits = self.lm_head(vt.view(-1, self.hidden_size))
            # vt = F.softmax((logits/self.gamma), dim=1)
            print("vt shape:", vt.shape)

        # Now pass in the previous character and get a new one
        # for ii in range(size):
        #     char, h = self.predict(words[-1], h, cuda=cuda, top_k=top_k)
        #     words.append(char)

        text = ['' for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for word in words[i]:
                text[i] = text[i] + ' ' + word
        return text

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        # self.lm_head.bias.data.fill_(0)
        # FC weights as random uniform
        self.lm_head.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # num_layers times 2 as it is bidirectional
        h = (weight.new(self.num_layers*2, n_seqs, int(self.hidden_size//2)).zero_(),
                  weight.new(self.num_layers*2, n_seqs, int(self.hidden_size//2)).zero_())
        self.h = h
        return h