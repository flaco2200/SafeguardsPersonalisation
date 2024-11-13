import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

# Path to dataset
race = "asian"
age = "u18"
gender = "nonbinary"
lgbtq = "othersex"
parent = "noparent"
pol = "otherpol"
rel = "veryrel"
race = "white"
#csv_dataset_path = f"datasets/{race}_dataset.csv"
csv_dataset_path = f"../datasets/race_datasets/{race}_dataset.csv"
#csv_labels = ["offensive"]
csv_labels = ["toxic_score"]

# Load the BERT tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class ToxicCommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_column = labels
        #self.text_column = 'response'
        self.text_column = "comment"

        # Ensure labels are numeric and fill missing values with 0
        self.dataframe[self.labels_column] = self.dataframe[self.labels_column].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.dataframe[self.labels_column] = self.dataframe[self.labels_column].astype(float)

        # Precompute the tokenization
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for i in range(len(self.dataframe)):
            comment = self.dataframe.iloc[i][self.text_column]
            labels = torch.tensor(self.dataframe.iloc[i][self.labels_column].tolist(), dtype=torch.float32)
            
            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            self.input_ids.append(encoding['input_ids'].flatten())
            self.attention_masks.append(encoding['attention_mask'].flatten())
            self.labels.append(labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_masks[index],
            'labels': self.labels[index]
        }
    
def load_csv_dataset(file_path):
    """ Load a CSV dataset. """
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_csv_dataset(csv_dataset_path)
    labels = csv_labels

    # Preprocess the dataset
    MAX_LEN = 128
    seed = 42  # You can change this seed to vary the splitting

    # Split the dataframe into training and testing sets with stratified sampling
    _, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    # Create datasets for training and testing
    test_dataset = ToxicCommentDataset(test_df, roberta_tokenizer, MAX_LEN, labels)

    # Save the preprocessed tensors for both training and testing
    test_preprocessed_data = {
        'input_ids': torch.stack(test_dataset.input_ids),
        'attention_mask': torch.stack(test_dataset.attention_masks),
        'labels': torch.stack(test_dataset.labels)
    }

    torch.save(test_preprocessed_data, f"preprocessed_{race}1_test_dataset.pt")