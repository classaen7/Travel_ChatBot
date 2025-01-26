from moviepy import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr



def extract_audio_from_video(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ko-KR')
    return text

def split_audio(audio_path, segment_length_ms):
    audio = AudioSegment.from_wav(audio_path)
    segments = []
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        segments.append(segment)
    return segments

def audio_segment_to_text(audio_segment):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_segment.export(format="wav")) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='ko-KR')
        except sr.UnknownValueError:
            text = "[인식 불가]"
        except sr.RequestError as e:
            text = f"[요청 오류: {e}]"
    return text

def process_audio_file(audio_path, segment_length_sec=60):
    segment_length_ms = segment_length_sec * 1000
    segments = split_audio(audio_path, segment_length_ms)
    all_text = []
    for idx, segment in enumerate(segments):
        print(f"Processing segment {idx + 1}/{len(segments)}...")
        text = audio_segment_to_text(segment)
        all_text.append(text)
    return "\n".join(all_text)

def video2text(video_path):
    aud_path_temp = './temp.wav'
    extract_audio_from_video(video_path, aud_path_temp)
    result_text = process_audio_file(aud_path_temp)
    
    return result_text