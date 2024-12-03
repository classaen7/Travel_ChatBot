# import spacy
from sentence_transformers import SentenceTransformer
import semchunk

# chunker = semchunk.chunkerify('gpt-4', 128)

def init_models(spc = "ko_core_news_md", stf = 'nlpai-lab/KoE5'):
    
    # nlp = spacy.load(spc)

    nlp = semchunk.chunkerify('nlpai-lab/KoE5', 128)
    embedder = SentenceTransformer(stf)

    return nlp, embedder

