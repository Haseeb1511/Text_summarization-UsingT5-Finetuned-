from huggingface_hub import create_repo
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 1: Create the repo (replace with your actual Hugging Face username)
# create_repo(repo_id="Haseeb1511/t5-finetune-text-summarization", private=False)

# Step 2: Load your model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./model")
tokenizer = T5Tokenizer.from_pretrained("./tokenizer")

# Step 3: Push them to Hugging Face
model.push_to_hub("Haseeb1511/t5-finetune-text-summarization")
tokenizer.push_to_hub("Haseeb1511/t5-finetune-text-summarization")
