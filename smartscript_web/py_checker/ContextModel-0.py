#!/usr/bin/env python
# coding: utf-8

# In[1]:



import time
import os
import itertools
from collections import defaultdict
import pickle
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from . import tokenization


# In[ ]:




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.NLLLoss(reduction='mean')
encoder_n_layers = 1
decoder_n_layers = 1
batch_size = 128
p_dropout = 0.1
model_size = 512
n_epoch = 200
vocab_size = 20000
lr = 0.0001
model_folder = "./model_type"
MAX_SEQ_LEN = 256
if not os.path.exists(model_folder):
     os.makedirs(model_folder)

# In[ ]:


# trainWriter = open("./TrainLog.txt","w",encoding='utf-8')
# validWriter = open("./ValidLog.txt","w",encoding='utf-8')
# testWriter = open("./TestLog.txt","w",encoding='utf-8')


# In[ ]:


spmPath = "/home/smartscript/smartscript_web/py_checker/ContextModel/"
# dataPath = "/home/Think/pkldata/"
dataPath = "/mnt/data/newdata/dict128.pkl"
# dataPath = '/media/siwei/Work/Google Cloud Backup/GPU/data/newdata128/dataset128.pkl'
Type500 = "/home/smartscript/smartscript_web/py_checker/ContextModel/"
max_tensor_length = 1000

# In[ ]:



with open (Type500+'Types500.pkl', 'rb') as fp:
    Types50 = pickle.load(fp)


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True)
        self.n_directions = 2

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        hidden = hidden.view(self.n_layers, self.n_directions, -1, self.hidden_size)
        hidden = hidden[-1, 0, :, :] + hidden[-1, 1, :, :]
        # Return output and final hidden state
        return outputs, hidden.unsqueeze(0)



class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                       encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, attn_mask=None):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        if attn_mask is not None:
            attn_energies.masked_fill_(attn_mask, -1e20)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)



class AttnClassifier(nn.Module):
    def __init__(self,
                 attn_model,
                 embedding,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 dropout=0.1):
        super(AttnClassifier, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, encoder_hidden, encoder_outputs, attn_mask):
        # Calculate attention weights from the current GRU output
        # attn_weights = self.attn(encoder_hidden, encoder_outputs, attn_mask)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector

        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        output = encoder_hidden.squeeze(0)
        # context = context.squeeze(1)
        # concat_input = torch.cat((output, context), 1)
        # concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output

class BugDetector(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 model_size=32,
                 p_dropout=0.1):
        super(BugDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_size, padding_idx=0)
        self.max_seq_len = max_seq_len
        self.encoder = EncoderRNN(model_size, self.embedding, encoder_n_layers, p_dropout)
        self.cls = AttnClassifier('dot', self.embedding, model_size, 500, decoder_n_layers, p_dropout)
        # self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seqs, seqs_lens):
        # Ignore the last EOS token
        encoder_outputs, encoder_hidden = self.encoder(seqs, seqs_lens)
        attn_mask = padding_mask(seqs_lens, self.max_seq_len)
        output = self.cls(encoder_hidden, encoder_outputs, attn_mask)
        return output


# In[ ]:

def load_data():
    global max_tensor_length
    sp = tokenization.load_model(spmPath+'spm.model')
    with open (dataPath, 'rb') as fp:
        alldata = pickle.load(fp)
    print("Init data len ",len(alldata))
    alldata = [x for x in alldata if len(x[0])<max_tensor_length]
    print("Final data len ",len(alldata))

    # max_tensor_length = 0
    
    # samples = []
    # labels = []
    # print("Loading data...")
    # for dt in tqdm(alldata[:5000]):
    # # for dt in alldata:
    #     dile = tokenization.encode(sp,'\n####\n')
    #     before = dt[0]
    #     curline = dt[1]
    #     after = dt[2]
    #     Name = dt[4]
    #     typeinfo = dt[3]
    #     if typeinfo =='?' or typeinfo == 'None' or typeinfo not in Types50:
    #         continue
        
    #     token_ids = tokenization.encode(sp, Name)
    #     if len(token_ids) == 0 or len(token_ids) > MAX_SEQ_LEN:
    #         continue
    #     before_emb = tokenization.encode(sp, before)
    #     curline_emb = tokenization.encode(sp, curline)
    #     after_emb = tokenization.encode(sp, after)
        
    #     labels.append(Types50.index(typeinfo))
    #     samples.append(before_emb+dile+curline_emb+dile+after_emb+dile+token_ids)
    #     len_max = max(len(before_emb),max(len(curline_emb),max(len(after_emb),len(token_ids))))
    #     max_tensor_length = max(max_tensor_length, len_max)
    return alldata, max_tensor_length


# In[ ]:



def padding_mask(seqs_lens, max_len):
    mask = torch.zeros((seqs_lens.size(0), seqs_lens.max().item()), dtype=torch.uint8)
    for i, seq_len in enumerate(seqs_lens):
        mask[i][seq_len:] = 1
    return mask.to(device)



def zero_padding(batch, fillvalue=0):
    batch.sort(key=lambda sample: len(sample[0]), reverse=True)
    batch_samples, batch_labels = zip(*batch)
    lengths = torch.tensor([len(indexes) for indexes in batch_samples])
    # return list(zip(*itertools.zip_longest(*batch_samples, fillvalue=fillvalue))), lengths, batch_labels
    # samples shape becomes: [max_len, batch_size]
    return list(itertools.zip_longest(*batch_samples, fillvalue=fillvalue)), lengths, batch_labels


def collate_fn(batch):
    padded_samples, lengths, batch_labels = zero_padding(batch, 0)
    return torch.LongTensor(padded_samples), torch.LongTensor(lengths), torch.LongTensor(batch_labels)


def compute_loss(pred, tgt):
    loss = criterion(pred, tgt)
    pred = pred.max(dim=1)[1]  # result of 'max' is tuple, dimension 1 is the indices, dimension 0 is the values
    n_correct = pred.eq(tgt).sum().item()
    return loss, n_correct


# In[ ]:



def train_epoch(model, training_data, optimizer):
    model.train()
    total_correct = 0
    total = 0
    for batch in tqdm(
            training_data, mininterval=2, desc=' ---Training--- ',
            leave=False):
        seqs, seqs_lens, labels = map(lambda x: x.to(device), batch)
        # optim.optimizer.zero_grad()
        optimizer.zero_grad()
        pred = model(seqs, seqs_lens)
        
        loss, n_correct = compute_loss(pred, labels)
        # loss.register_hook(lambda grad: print(grad))
        loss.backward()
        optimizer.step()
        # max_norm(model, 3)
        total += labels.size(0)
        total_correct += n_correct
    accr = total_correct / total
    return accr


# In[ ]:




def eval_epoch(model, validation_data):
    model.eval()  # disable dropout, batchnorm, etc.
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc=' ---Validation--- ',
                          leave=False):
            seqs, seqs_lens, labels = map(lambda x: x.to(device),
                                          batch)
            pred = model(seqs, seqs_lens)
            _, n_correct = compute_loss(pred, labels)
            total += labels.size(0)
            total_correct += n_correct
    accr = total_correct / total
    return accr


# In[ ]:



def test_epoch(model, test_data):
    model.eval()  # disable dropout, batchnorm, etc.
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                          desc=' ---Test--- ',
                          leave=False):
            seqs, seqs_lens, labels = map(lambda x: x.to(device),
                                          batch)
            pred = model(seqs, seqs_lens)
            _, n_correct = compute_loss(pred, labels)
            total += labels.size(0)
            total_correct += n_correct
    accr = total_correct / total
    return accr


# In[ ]:



def train(model, training_data, validation_data, test_data, optim, vocab_size, max_tensor_length):
    val_accrs = []
    test_accrs = []
    for i in range(n_epoch):
        start = time.time()
        train_accr = train_epoch(model, training_data, optim)
        trainWriter.write(str(train_accr)+"\n")
        # trainWriter.write("\n")
        trainWriter.flush()
        print('\n  - (Training)   accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
            accu=100 * train_accr,
            elapse=(time.time() - start) / 60))
        start = time.time()
        val_accr = eval_epoch(model, validation_data)
        validWriter.write(str(val_accr)+"\n")
        # validWriter.write("\n")
        validWriter.flush()
        print('\n  - (Validation)   accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
            accu=100 * val_accr,
            elapse=(time.time() - start) / 60))
        val_accrs.append(val_accr)
        # print("Accuracies so far: ", val_accrs)
        
        
        start = time.time()
        test_accr = test_epoch(model, test_data)
        testWriter.write(str(test_accr)+"\n")
        # validWriter.write("\n")
        testWriter.flush()
        print('\n  - (Test)   accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
            accu=100 * test_accr,
            elapse=(time.time() - start) / 60))
        test_accrs.append(test_accr)
        # print("Accuracies so far: ", val_accrs)
        
        
        model_state_dict = model.state_dict()
        config = {'max_src_seq_len': max_tensor_length,
                  'vocab_size': vocab_size,
                  'dropout': p_dropout}
        checkpoint = {'model': model_state_dict, 'epoch': i,
                      'config': config}
        model_name = os.path.join(model_folder, "TypeModel.ckpt")
        if val_accr >= max(val_accrs):
            print("Save model at epoch ", i)
            torch.save(checkpoint, model_name)


# In[ ]:



def main():
    samples, max_tensor_length = load_data()
    print(len(samples))
    training_samples, validation_samples, test_samples = random_split(
        samples, [int(len(samples) * 0.6), int(len(samples) * 0.2),len(samples) - int(len(samples) * 0.6)-int(len(samples) * 0.2)])
    train_loader = torch.utils.data.DataLoader(
        training_samples,
        num_workers=0,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True)
    valid_loader        = torch.utils.data.DataLoader(
        validation_samples,
        num_workers=0,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    test_loader        = torch.utils.data.DataLoader(
        test_samples,
        num_workers=0,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    # vocab size should be len(word2index)+1 since 0 is not used
    detector = BugDetector(vocab_size, max_tensor_length, model_size, p_dropout)
    optimizer = optim.Adam(detector.parameters(), lr=lr, weight_decay=0.001)
    detector.to(device)
    train(detector, train_loader, valid_loader,test_loader, optimizer, vocab_size, max_tensor_length)


# In[ ]:


if __name__=="__main__":
    main()

