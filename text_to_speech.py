import edge_tts

async def text_to_speech(text, output_file):
    """
    Convert text to speech using Edge TTS and save it to an output file.
    
    Args:
        text (str): The text to convert to speech.
        output_file (str): The path to the output audio file.
    """
    communicate = edge_tts.Communicate(text, voice='en-GB-LibbyNeural', rate='+15%')
    await communicate.save(output_file)
    print(f"Audio saved to {output_file}")
    
    

