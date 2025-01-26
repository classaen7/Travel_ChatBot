from sentence_transformers import SentenceTransformer
import semchunk



def init_models(spc = "ko_core_news_md", stf = 'nlpai-lab/KoE5'):
    """
    RAG 검색 시스템 구축을 위한 모델 선언
    Semchunk와 SentenceTransformer
    """
    
    nlp = semchunk.chunkerify('nlpai-lab/KoE5', 128)
    embedder = SentenceTransformer(stf)

    return nlp, embedder

