# Encoder vs Decoder: Multilingual BERT and Monolingual GPT-2

One Streamlit app with two separate sections.

**Live app:** PASTE_YOUR_STREAMLIT_LINK_HERE

| Section | Model | Type | Language | Dataset | Task |
|---|---|---|---|---|---|
| Encoder | BERT (`bert-base-multilingual-cased`) | Encoder-only | Multilingual (104) | IMDB movie reviews | Sentiment classification |
| Decoder | GPT-2 (`gpt2`) | Decoder-only | Monolingual (English) | WikiText-2 | Text generation |

- Encoder model: https://huggingface.co/TahiraAbbas/bert-multilingual-imdb-sentiment
- Decoder model: https://huggingface.co/TahiraAbbas/gpt2-wikitext

## Files

- `app.py`: Streamlit app (encoder section and decoder section)
- `requirements.txt`: Python packages
- `sample_reviews.csv`: 30 IMDB reviews used as examples in the encoder section
- `Encoder_BERT_Colab.ipynb`, `Decoder_GPT2_Colab.ipynb`: fine-tuning code

## Run locally

    pip install -r requirements.txt
    streamlit run app.py
