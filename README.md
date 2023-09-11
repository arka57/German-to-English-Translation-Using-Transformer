# German-to-English-Translation-Using-Transformer
A Seq2Seq german to english translation using Transformer

Using vanilla transformer a seq2seq german to english translator was developed using inbuilt transfomer module of nn.Transformer(
).

# Dataset
Used Multi30k dataset for model training. It contains german sentences and its corresponding english sentences

# Prerequisites
Install the required pip packages:
pip install -r requirements.txt
Install spacy models :
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm


Train
python train.py

#Inference exapmles<br>

<img width="613" alt="t1" src="https://github.com/arka57/German-to-English-Translation-Using-Transformer/assets/36561428/3072a143-88b7-4fad-a7c3-c8583870dcf3">

<img width="641" alt="t2" src="https://github.com/arka57/German-to-English-Translation-Using-Transformer/assets/36561428/23b0f6ab-9fb4-4a31-97c5-c8a90a4a1514">
