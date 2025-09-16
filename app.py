import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import os
import pyttsx3

from detector import DrowsinessDetector

# Initialize your detector
detector = DrowsinessDetector()

# Setup TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def play_alert_sound():
    engine.say("Video is stopped")
    engine.runAndWait()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alerted_last = False  # track last frame alert state

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        output_frame, alert_flag = detector.process_frame(img)

        # If alert_flag is True this frame, play sound
        # and optionally ensure it happens each frame or only on transitions
        # Here each time alert_flag is True
        if alert_flag:
            play_alert_sound()

        # Convert back to av.VideoFrame
        new_frame = av.VideoFrame.from_ndarray(output_frame, format="bgr24")
        return new_frame

def main():
    st.title("Drowsiness Detection Web App")

    ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.write("If your eyes stay closed beyond threshold, this app will show a visual alert and play a sound.")

if __name__ == "__main__":
    main()
