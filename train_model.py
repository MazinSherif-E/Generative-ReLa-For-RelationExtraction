import argparse
import pandas as pd
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq)
import numpy as np
import nltk
import datasets
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Define your functions here
def create_data_frame(file_path):
      with open(file_path, "r") as file:
          data = file.readlines()

      relations = []
      sentences = []

      for line in data:
        parts = line.strip().split(" ")
        relation = parts[1]
        sentence = " ".join(parts[3:])
        relations.append(relation)
        sentences.append(sentence)

      return pd.DataFrame({"Relation": relations, "Sentence": sentences})

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["Sentence"], batch["Relation"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

def postprocess_text(preds, labels):
    if preds is None or labels is None:
        print("Warning: Either preds or labels is None.")
        return [], []

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def get_f1(key, prediction, none_id):
    correct_by_relation = ((key == prediction) & (prediction != none_id)).astype(np.int32).sum()
    guessed_by_relation = (prediction != none_id).astype(np.int32).sum()
    gold_by_relation = (key != none_id).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

def get_score():
    
    pred = trainer.predict(
            batch_test_data, metric_key_prefix="predict", max_length=32, num_beams=4
        )

    label_seq2seq = []
    pred_seq2seq = []
    
    print('start generate pred_seq2seq')

    for k, d in tqdm(enumerate(batch_test_data)):
        
        tt = d['labels']
        temp_label = tokenizer.decode(tt[:np.sum(np.array(tt) != -100)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        temp_pred = tokenizer.decode(pred[0][k], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label_seq2seq.append(temp_label)
        pred_seq2seq.append(temp_pred)

    print('*****finish predict*****')
    def func(x):
        if x in label_seq2seq:
            return x
        else:
            return 'Other'
        
    pred_seq2seq = [func(x) for x in pred_seq2seq]
    
    df = pd.DataFrame()

    df['label'] = label_seq2seq
    df['pred'] = pred_seq2seq
    print('*****finish df*****')
    
    lb = LabelEncoder()
    lb.fit(list(df['label']))
    label_lb = lb.transform(list(df['label']))
    pred_lb = lb.transform(list(df['pred']))

    print('*****finish encode*****')

    P, R, F1 = get_f1(label_lb, pred_lb, lb.transform(['Other'])[0])
    
    return P, R, F1

augmentations = {
    "Cause-Effect": ["reason-result", "consequence", "led to", "outcome of", "causation", "impact"],
    "Component-Whole": ["part-whole", "element", "make up", "composed of", "constituent", "component parts"],
    "Content-Container": ["substance-holding", "material", "contains", "stored in", "contents", "containerized"],
    "Entity-Destination": ["object-place", "target", "headed to", "destination of", "entity's final stop", "endpoint"],
    "Entity-Origin": ["source-entity", "origin", "came from", "origin of", "entity's starting point", "birthplace of"],
    "Instrument-Agency": ["tool-agent", "means", "used by", "agency behind", "instrumentality", "mechanism"],
    "Member-Collection": ["individual-group", "part", "belongs to", "collection of", "member", "set of"],
    "Message-Topic": ["communication-subject", "theme", "talks about", "topic of", "message content", "subject matter"],
    "Product-Producer": ["outcome-maker", "creation", "made by", "producer of", "product's origin", "manufacturer"]
}

def augment_data(data):
    augmented_data = []

    for index, row in data.iterrows():
        relation = row['Relation'].split('(')[0]
        sentence = row['Sentence']
        
        if relation not in augmentations:
            continue

        augmentations_for_relation = augmentations.get(relation)

        for augmentation in augmentations_for_relation:
            augmented_data.append({"Sentence": sentence, "Relation": augmentation + "(e2,e1)"})
    
    return pd.DataFrame(augmented_data)

def add_augmentations(data, subset_size):
    subset_size = int(subset_size * len(data))
    subset_data = train_data.sample(n=subset_size, random_state=42)
    augmented_subset_data = augment_data(subset_data)
    combined_train_data = pd.concat([data, augmented_subset_data])
    shuffled_train_data = combined_train_data.sample(frac=1, random_state=42)
    return shuffled_train_data.reset_index(drop=True)


def main(
    train_file, 
    test_file, 
    validation_file, 
    use_augmentation, 
    subset_size, 
    model_name, 
    save_dir, 
    save_name, 
    epochs, 
    batch_size, 
    lr, 
    warmup_ratio, 
    label_smoothing_factor, 
    saved_steps):

    # Load data from file paths
    train_data = create_data_frame(train_file)
    test_data = create_data_frame(test_file)
    validation_data = create_data_frame(validation_file)

    # Augment data if specified
    if use_augmentation:
        train_data = add_augmentations(train_data, subset_size=subset_size)
        validation_data = add_augmentations(validation_data, subset_size=subset_size)

    # Tokenize and preprocess data
    encoder_max_length = 128
    decoder_max_length = 32
    batch_train_data = train_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )
    batch_validation_data = validation_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=validation_data.column_names,
    )
    batch_test_data = test_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=test_data.column_names,
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        evaluation_strategy='epoch',
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=label_smoothing_factor,
        predict_with_generate=True,
        logging_dir="logs",
        logging_steps=200,
        report_to='none',
        save_total_limit=1,
    )
    
    # Compute metrics
    nltk.download("punkt", quiet=True)
    metric = datasets.load_metric("rouge")
    
    # Define trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=batch_train_data,
        eval_dataset=batch_validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()

    # Compute and return score
    P, R, F1 = get_score()
    print('{}:, P:{}, R:{}, F1 :{} \n'.format('score', P, R, F1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument('-f')

    # Data file paths and augmentation arguments
    parser.add_argument("train_file", help="Path to the training data file")
    parser.add_argument("test_file", help="Path to the test data file")
    parser.add_argument("validation_file", help="Path to the validation data file")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--subset_size", type=float, default=0.5, help="Subset size for augmentation (0.25 or 0.5)")

    # Model configuration arguments
    parser.add_argument("--model_name", default="bart-base", type=str,
                        help="in [bart-base, bart-large, bart-base-cnn, bart-large-cnn, t5-small, t5-base, t5-large, t5-3b, t5-11b]")
    parser.add_argument("--save_dir", default="dir", type=str)
    parser.add_argument("--save_name", default="bart-base-v1", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.2, type=float)
    parser.add_argument("--label_smoothing_factor", default=0.1, type=float)
    parser.add_argument("--saved_steps", default=2000, type=int)

    args = parser.parse_args()

    main(args.train_file, args.test_file, args.validation_file, args.use_augmentation, args.subset_size,
         args.model_name, args.save_dir, args.save_name, args.epochs, args.batch_size, args.lr,
         args.warmup_ratio, args.label_smoothing_factor, args.saved_steps)