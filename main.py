from predict import AudioPredictor

if __name__ == "__main__":
    # Model config
    model_path = 'Models/v2-disease-predict.keras'
    predictor = AudioPredictor(model_path)
    
    # Audio config
    audio_path = 'Test/1. Bronchiectasis/116_1b2_Pl_sc_Meditron.wav'
    result = predictor.predict_audio(audio_path)
    
    print(f"This model predicts the audio as: {audio_result['audio_label']} with prediction scores: {audio_result['prediction_score']}")