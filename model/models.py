# import spacy
from sentence_transformers import SentenceTransformer
import semchunk

# chunker = semchunk.chunkerify('gpt-4', 128)

def init_models(spc = "ko_core_news_md", stf = 'all-MiniLM-L6-v2'):
    
    # nlp = spacy.load(spc)

    nlp = semchunk.chunkerify('gpt-4', 64)
    embedder = SentenceTransformer(stf)

    return nlp, embedder

