import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import nltk
import torch
from sklearn.metrics import classification_report
import sys

def identify_metaphor_sentence(txt, metaphor):
    for sentence in nltk.sent_tokenize(txt):
        if metaphor in sentence:
            return sentence

nltk.download('punkt')

def load_data_and_model(data):
    df = pd.read_csv(data)

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])

    # Replace metaphor IDs with actual words
    metaphor = {0:'road', 1:'candle', 2:'light', 3:'spice', 4:'ride', 5:'train', 6:'boat'}
    df.replace({"metaphorID": metaphor}, inplace=True)

    # Extract sentence with metaphor
    df['text'] = df.apply(lambda x: identify_metaphor_sentence(x['text'], x['metaphorID']), axis=1)
    df = df.rename(columns={'metaphorID': 'metaphor_word'})

    # Drop rows where metaphor sentence was not found
    df = df.dropna(subset=['text', 'label_boolean'])

    # Train-test split
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Tokenize (sentence + metaphor word as pair)
    train_tokens = tokenizer(list(train_data["text"]), list(train_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")
    test_tokens = tokenizer(list(test_data["text"]), list(test_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")

    # Create TensorDatasets
    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(list(train_data["label_boolean"].astype(int)))
    )

    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(list(test_data["label_boolean"].astype(int)))
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_data, test_data, train_loader, test_loader, model

def train(train_data, test_data, train_loader, test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    early_stopping_patience = 5
    best_loss = float('inf')
    current_patience = 0

    losses = []
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} with best loss {best_loss:.4f}")
                break

    # Evaluation
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_report = classification_report(true_labels, predictions)
    predictions_df = test_data.copy()
    predictions_df.insert(3, 'Predictions', predictions)
    predictions_df.iloc[:, 3] = predictions_df.iloc[:, 3].astype(bool)
    predictions_df = predictions_df.iloc[:, [0, 2, 1, 3]]

    return losses, model, test_report, predictions_df

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <data.csv>")
        sys.exit(1)

    file_name = sys.argv[1]
    train_data, test_data, train_loader, test_loader, model = load_data_and_model(file_name)
    losses, model, test_report, predictions_df = train(train_data, test_data, train_loader, test_loader, model)

    print(test_report)
    print(predictions_df)
    model.save_pretrained('distilbert_metaphor_model')

if __name__ == "__main__":
    main()
