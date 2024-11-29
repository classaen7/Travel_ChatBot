import gradio as gr
import numpy as np
from PIL import Image
import base64
from io import BytesIO

from RAG.rag import RAGSystem
from RAG.video2text import video2text
from model.models import init_models

# def PDF 처리

# Init GLOBAL
NLP_MODEL, EMB_MODEL = init_models()
RAG = RAGSystem(NLP_MODEL, EMB_MODEL)
   

def pdf_change(pdf_path):
    if pdf_path is not None:
        pdf_chunks = RAG.process_pdf(pdf_path)
        RAG.create_index(pdf_chunks)
    return None


def video_change(video_path):
    if video_path is not None:
        video_text = video2text(video_path)
        video_chunks = RAG.process_video(video_text)
        RAG.create_index(video_chunks)
        
    return None


def run_text_inference(question, chat_history):
        
        # Prepare messages from chat history
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

        # Add the current question as the latest message from the user
        messages.append({"role": "user", "content": question})

        # # Query vLLM with the prepared messages structure
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     max_tokens=512,
        # )

        # Extract the response content for Gradio to display
        # answer = response.choices[0].message.content
        return "answer"
    
def respond(message, image, chat_history):

    bot_message = run_text_inference(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    import time
    time.sleep(2)
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


        msg.submit(respond, [msg, image_input, chatbot], [msg, chatbot])
        


    # 인터페이스 실행
    demo.launch()


if __name__ == "__main__":
    main()