# from ...

if __name__ == "__main__":
    # Model config
    model_path = 'Models/v2-disease-predict.keras'
    predictor = AudioPredictor(model_path)
    
    # Audio config
    audio_path = 'Test/1. Bronchiectasis/116_1b2_Pl_sc_Meditron.wav'
    result = predictor.predict_audio(audio_path)
    
    print("{audio_label}")