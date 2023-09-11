import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from transformer import Transformer



spacy_ger=spacy.load('de_core_news_sm')
spacy_en=spacy.load('en_core_web_sm')


#Tokenizing
def tokenize_ger(text):
    return [tokens.text for tokens in spacy_ger.tokenizer(text)]

def tokenize_en(text):
    return [ tokens.text for tokens in spacy_en.tokenizer(text)]


german=Field(tokenize=tokenize_ger,lower=True,init_token="<sos>",eos_token="<eos>")
english=Field(tokenize=tokenize_en,lower=True,init_token="<sos>",eos_token="<eos>")

#Making Train Test Validation Split

train_data,validation_data,test_data=Multi30k.splits(
    exts=(".de",".en"),fields=(german,english),root='data'
    )

#Building Vocab
german.build_vocab(train_data,max_size=10000,min_freq=2)
english.build_vocab(train_data,max_size=10000,min_freq=2)



# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train():
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    sentence = "ein pferd geht unter einer brücke neben einem boot."

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            checkpoint={
                "state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict()
            }
            save_checkpoint(checkpoint)


        model.eval()
        translated_sentence=translate_sentence(
            model,sentence,german,english,device,max_length=50
        )    
        print(f"Translated example sentence: \n {translated_sentence}")
        model.train()
        losses=[]

        for batch_idx,batch in enumerate(train_iterator):
            #get input and targets and get to cuda
            inp_data=batch.src.to(device)
            target=batch.trg.to(device)

            #Forward Prop
            output=model(inp_data,target[:-1,:])


            output=output.reshape(-1,output.shape[2])
            target=target[1:].reshape(-1)

            optimizer.zero_grad()

            loss=criterion(output,target)
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)

            optimizer.step()

            if batch_idx%100==0:
                print("Training loss",loss.item())

        mean_loss=sum(losses)/len(losses)
        scheduler.step(mean_loss)

    score=bleu(test_data[1:100],model,german,english,device)
    print(f"Bleu score {score * 100:.2f}")        



def predict():
    
    load_checkpoint(torch.load("my_checkpoint.pth.tar",map_location=torch.device('cpu')), model, optimizer)
 
    sentence=input("enter sentence:")
    model.eval()
    translated_sentence=translate_sentence(
        model,sentence,german,english,device,max_length=50)    
    print(f"Translated example sentence: \n {translated_sentence}")
