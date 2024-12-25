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
    text = """
ایک دور دراز گاؤں میں ایک ننھا شہزادہ رہتا تھا جس کا نام ارحم تھا۔ ارحم نہایت ذہین، بہادر اور دل کا نرم تھا۔ گاؤں کے لوگ اس سے بہت محبت کرتے تھے کیونکہ وہ ہمیشہ دوسروں کی مدد کے لئے تیار رہتا تھا۔

ایک دن گاؤں کے قریب ایک خوفناک دیو آگیا جو گاؤں کے کھیتوں اور گھروں کو تباہ کر رہا تھا۔ گاؤں کے لوگ خوف زدہ تھے اور کوئی بھی دیو کا سامنا کرنے کو تیار نہ تھا۔ بزرگوں نے کہا کہ صرف کسی بہادر شخص کی قربانی ہی دیو کو روک سکتی ہے۔

ارحم نے یہ سنا تو اس کے دل میں گاؤں کو بچانے کا عزم پیدا ہوا۔ وہ اپنے والدین کے پاس گیا اور ان سے کہا، "مجھے دیو کا سامنا کرنا ہے تاکہ گاؤں محفوظ ہو جائے۔" والدین کے آنکھوں میں آنسو تھے لیکن انہوں نے اپنے بیٹے کے عزم کو سلام کیا۔

ارحم نے اپنی تلوار اٹھائی اور دیو کے غار کی طرف روانہ ہو گیا۔ راستے میں اس نے خوب دعائیں کیں اور اپنے اللہ پر بھروسہ رکھا۔ جب وہ دیو کے سامنے پہنچا تو دیو ہنسا اور کہا، "ایک ننھا لڑکا میری طاقت کا مقابلہ کرے گا؟"

لیکن ارحم نے ہمت نہیں ہاری۔ اس نے اپنے دل کی طاقت سے دیو کا مقابلہ کیا۔ آخر کار دیو کو شکست ہوئی اور وہ غائب ہو گیا۔ ارحم نے گاؤں کو بچا لیا۔

جب وہ واپس آیا تو گاؤں کے لوگ خوشی سے جھوم اٹھے۔ ارحم کی بہادری کی کہانی ہر طرف مشہور ہو گئی اور وہ گاؤں کے ہیرو بن گیا۔
"""
    # Call the client function
    tts_client(api_url, text)
