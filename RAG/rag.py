import gradio as gr
import faiss
import numpy as np
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from vllm.sampling_params import SamplingParams


# RAG(Retrieval-Augmented Generation) 시스템의 핵심 클래스
# LLM(언어 모델), NLP 처리기, 임베딩 모델을 통합하여 관리
class RAGSystem:
    def __init__(self, nlp, embedder):
        self.nlp = nlp          # 자연어 처리 모델
        self.embedder = embedder # 텍스트 임베딩 모델
        self.index = None       # FAISS 검색 인덱스
        self.chunks = []        # 텍스트 청크 저장소

        self._init_index()


    def _init_index(self):
        emb_dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(emb_dim)

    # PDF 문서를 처리하여 문장 단위로 분할하는 메서드
    def process_pdf(self, pdf_path: str) -> List[str]:
        chunks = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            page_chunks = self.nlp(text)
            chunks.extend(page_chunks)
        
        return chunks

    def process_video(self, text: str) -> List[str]:
        chunks = self.nlp(text)
        
        return chunks


    # 텍스트 청크들의 벡터 인덱스를 생성하는 메서드
    def create_index(self, chunks: List[str]):
        if not chunks:
            return

        self.chunks.extend(chunks)

        # 텍스트를 벡터로 변환
        embeddings = self.embedder.encode(chunks)
        self.index.add(embeddings.astype('float32'))
        
        print(f"인덱스 생성 완료: {len(chunks)}개의 청크")


    # 질문과 관련된 청크들을 검색하는 메서드
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        if not self.index:
            return []

        # 질문을 벡터화하고 가장 유사한 k개의 청크 검색
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.chunks[idx] for idx in indices[0]]

