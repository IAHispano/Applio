import requests
import sounddevice as sd
import wave
from io import BytesIO
import numpy as np

def tts_client(api_url: str, text: str):
    """
    Sends text to the TTS API, receives audio, and plays it.
    
    Args:
        api_url (str): The URL of the TTS API endpoint.
        text (str): The text to synthesize into speech.
    """
    try:
        # Send text to the API
        response = requests.post(api_url, json={"tts_text": text})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Convert response bytes to a WAV file object
        audio_bytes = BytesIO(response.content)
        with wave.open(audio_bytes, "rb") as wav_file:
            # Extract audio parameters
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            audio_data = wav_file.readframes(wav_file.getnframes())

        # Convert audio data to NumPy array for playback
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Play audio using sounddevice
        print("Playing audio...")
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()  # Wait until playback is finished
        print("Audio playback finished.")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the TTS API: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the API URL
    api_url = "http://5.9.81.185:9033/tts"

    # Input text for TTS
    text = "یہ ایک ٹیسٹ پیغام ہے۔"

    # Call the client function
    tts_client(api_url, text)
