import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CHECKPOINT_PATH = "finetuned_twitter_roberta_multi.pt"
SAVE_DIR = "finetuned_twitter_roberta_multi"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the same architecture used in training
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 3)  # 3 outputs

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        x = self.dropout(cls_output)
        return self.regressor(x)

# Rebuild and load trained weights
model = RobertaRegressionHead()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))

# Save tokenizer + config
tokenizer.save_pretrained(SAVE_DIR)
model.roberta.config.save_pretrained(SAVE_DIR)

# Save model weights in Hugging Face format
torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")

print(f"✅ Saved full model + tokenizer to '{SAVE_DIR}/'")
