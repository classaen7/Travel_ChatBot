import spacy
import pytextrank

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_md")

# PyTextRank를 파이프라인에 추가
nlp.add_pipe("textrank")


def summary_text(hist):
    assistant_hist = []
    user_hist = []

    for msg in hist:
        if msg["role"] == "assistant":
            assistant_hist.append(msg["content"])
        
        elif msg["role"] == "user":
            user_hist.append(msg["content"])

    assistant_text = ' '.join(assistant_hist)
    user_text = ' '.join(user_hist)

    a_doc = nlp(assistant_text)
    u_doc = nlp(user_text)

    a_result = [
        sent.text for sent in a_doc._.textrank.summary(limit_phrases=15, limit_sentences=3)
    ]

    u_result = [
        sent.text for sent in u_doc._.textrank.summary(limit_phrases=15, limit_sentences=3)
    ]


    return ' '.join(a_result), ' '.join(u_result)