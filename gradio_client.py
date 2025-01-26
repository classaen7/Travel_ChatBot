import gradio as gr
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import time

from RAG.rag import RAGSystem
from RAG.video2text import video2text
from model.models import init_models
from openai import OpenAI
from RAG.summary_history import summary_text


# Init GLOBAL
NLP_MODEL, EMB_MODEL = init_models()
RAG = RAGSystem(NLP_MODEL, EMB_MODEL)

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def pdf_change(pdf_path):
    if pdf_path is not None:
        pdf_chunks = RAG.process_pdf(pdf_path)
        RAG.create_index(pdf_chunks)


def video_change(video_path):
    if video_path is not None:
        video_text = video2text(video_path)
        video_chunks = RAG.process_video(video_text)
        RAG.create_index(video_chunks)
        


system_prompt = """
당신은 전문 여행 가이드 챗봇입니다.
사용자에게 현지 명소, 맛집, 교통 정보, 숙박 정보 등 여행과 관련된 모든 정보를 제공합니다.
친근하고 이해하기 쉬운 어조를 사용하며, 명확한 문장으로 답변을 마무리합니다.
질문이 복잡할 경우, 단계별로 문제를 분석하고 각 단계를 설명하며 답변을 제공합니다.
사용자가 요청하지 않더라도 추가로 유용할 수 있는 정보를 적절히 제안합니다.
사용자가 요청한 정보가 간단할 경우, 바로 답변을 제공합니다.

예시:
- 사용자 질문: "이탈리아에서 유명한 관광지는?"
- 답변: "이탈리아에서 가장 유명한 관광지로는 로마의 콜로세움, 피렌체의 두오모 성당, 베네치아의 대운하 등이 있습니다. 특히, 콜로세움은 로마 제국의 역사를 느낄 수 있는 대표적인 장소입니다. 추가로 알고 싶은 도시가 있나요?"

위의 지침을 엄격히 따르며, 같은 말을 반복하지 마세요.
"""


def run_text_inference(questions, chat_history):
        
        if isinstance(questions["content"], list):
            query = questions["content"][1]["text"]
        else:
            query = questions["content"]


        # Vector Search
        rag_text = ""

        if RAG.index.ntotal > 0:

            query_embedding = EMB_MODEL.encode([query])
        
            k = 5 # 이웃 수
            distance_threshold = 0.5 # 거리 임계값

            distances, indices = RAG.index.search(query_embedding, k)
            
            
            for idx, i in enumerate(indices[0]):

                if distances[0][idx] < 0.85:
                    rag_text += RAG.chunks[i]

        if len(rag_text)>0:
            print(rag_text)
            rag_text = "참고할 맥락 : " + rag_text + "질문"
        else:
            print("---------------No RAG---------------")


        # RAG 메시지 추가
        if isinstance(questions["content"], list):
            questions["content"][1]["text"] = rag_text + questions["content"][1]["text"]
        else:
            questions["content"] = rag_text + questions["content"]


        messages = [
            {
                "role": "system",
                "content": system_prompt
            }]
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in chat_history])
        messages.append(questions)

        
        try:
            # Query vLLM with the prepared messages structure
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.1,  # 응답의 다양성 조절
                top_p=0.9         # 고려할 토큰 후보군의 범위 설정
            )
            
            
        except: # 텍스트 요약 처리
            messages = [
            {
                "role": "system",
                "content": system_prompt
            }]
            a,u = summary_text(chat_history)


            messages.append({"role":"user", "content" : u})
            messages.append({"role":"assistant", "content" : a})

            messages.append(questions)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.1,  # 응답의 다양성 조절
                top_p=0.9         # 고려할 토큰 후보군의 범위 설정
            )



        # Extract the response content for Gradio to display
        answer = response.choices[0].message.content
        return answer

        
def respond(message, image, chat_history,):

    def image_encode(image):
        image_pil = Image.fromarray(image)
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    if image is not None:
        image_base64 = image_encode(image)

        question = {"role": "user", "content" : [ {
            "type" : "image_url",
            "image_url" : {"url": f"data:image/jpeg;base64,{image_base64}"}  
        }, 
        {
            "type" : "text",
            "text" : message
        }]}
    
    else:
        question = {"role": "user", "content": message}
    
    
    bot_message = run_text_inference(question, chat_history)
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    
    time.sleep(1)
    
    return "", chat_history

def main():

    with gr.Blocks() as demo:
        gr.Markdown("# 여행 가이드 Chat BOT")
        
        with gr.Row():
            # PDF 업로드 컴포넌트
            pdf_input = gr.File(label="PDF 업로드", file_types=['.pdf'])
            
            # pdf 업로드 시 함수 호출
            pdf_input.change(
                fn=pdf_change,
                inputs=pdf_input
            )
            
            # 동영상 업로드 컴포넌트
            video_input = gr.Video(label="동영상 업로드")
            
            # 동영상 업로드 시 함수 호출
            video_input.change(
                fn=video_change,
                inputs=video_input
            )

        
        # 이미지 업로드 컴포넌트
        image_input = gr.Image(label="이미지 업로드")
        
        
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()

        clear = gr.ClearButton([msg, chatbot])  


        msg.submit(respond, [msg, image_input, chatbot], [msg, chatbot])
        
    

    # 인터페이스 실행
    demo.launch(share=True)


if __name__ == "__main__":
    main()