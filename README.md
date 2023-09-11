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
<img width="960" alt="t1" src="https://github.com/arka57/German-to-English-Translation-Using-Transformer/assets/36561428/1df1552a-0305-4bde-bfa9-d48e5dd3ad12">
<img width="960" alt="t2" src="https://github.com/arka57/German-to-English-Translation-Using-Transformer/assets/36561428/4f24a911-0914-40db-bc19-c759cddc505b">

