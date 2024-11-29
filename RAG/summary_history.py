import spacy
import pytextrank

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_md")

# PyTextRank를 파이프라인에 추가
nlp.add_pipe("textrank")


def summary_text(hist, limit_phrases=15, limit_sentences=3):
    doc = nlp(hist)
    result = [
        sent.text for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=3)
    ]
    return ' '.join(result)