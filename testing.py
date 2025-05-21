from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

def load_model(model_path):
    """Load the trained model and tokenizer"""
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_cost(tokenizer, model, state, district, year, season, crop):
    """Make a prediction"""
    text = (
        f"State: {state}, District: {district}, "
        f"Year: {year}, Season: {season}, Crop: {crop}"
    )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits.item()

def get_user_input():
    """Prompt user for prediction parameters"""
    print("\n" + "="*40)
    print("Crop Cost Prediction System")
    print("="*40)
    
    state = input("Enter State: ")
    district = input("Enter District: ")
    year = input("Enter Crop Year: ")
    season = input("Enter Season (Kharif/Rabi): ")
    crop = input("Enter Crop Name: ")
    
    return state, district, year, season, crop

def main():
    # Load model (update path if needed)
    MODEL_PATH = "./fast_crop_model"
    tokenizer, model = load_model(MODEL_PATH)
    
    while True:
        try:
            # Get user input
            state, district, year, season, crop = get_user_input()
            
            # Make prediction
            cost = predict_cost(tokenizer, model, state, district, year, season, crop)
            
            # Display result
            print("\nPrediction Result:")
            print("-"*40)
            print(f"State: {state}")
            print(f"District: {district}")
            print(f"Year: {year}")
            print(f"Season: {season}")
            print(f"Crop: {crop}")
            print("-"*40)
            print(f"PREDICTED COST: ₹{cost:.2f}")
            print("-"*40)
            
            # Continue?
            cont = input("\nPredict another? (y/n): ").lower()
            if cont != 'y':
                print("\nThank you for using the Crop Cost Predictor!")
                break
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with valid inputs.\n")

if __name__ == "__main__":
    main()
