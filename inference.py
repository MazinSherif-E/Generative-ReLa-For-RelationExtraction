import argparse
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # Define the available model directories
    model_directories = {
        "no_augmentations": "Mazin100/no_augmentation_relation_extractoin",
        "augmentations_25": "Mazin100/augmentation_25_relation_extractoin",
        "augmentations_50": "Mazin100/augmentation_50_relation_extractoin"
    }
    
    print("Available models:")
    for idx, model_name in enumerate(model_directories.keys()):
        print(f"{idx + 1}. {model_name}")
    
    while True:
        choice = input("Enter the number corresponding to the model you want to use: ")
        if choice.isdigit() and int(choice) in range(1, len(model_directories) + 1):
            break
        else:
            print("Invalid choice. Please enter a valid number.")
    
    chosen_model = list(model_directories.keys())[int(choice) - 1]
    
    # Load the chosen model
    model_directory = model_directories[chosen_model]
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
    
    # Set up the pipeline with the loaded model
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    # Prompt user to enter a sentence
    sentence = input("Enter the sentence for inference: ")
    
    # Perform inference
    preds = generator(sentence)
    
    # Display the result
    print("\nRelation:", preds[0]['generated_text'])

if __name__ == "__main__":
    main()
