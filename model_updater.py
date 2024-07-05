import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

class ModelUpdater:
    def __init__(self, model_path, special_tokens):
        self.model_path = model_path
        self.special_tokens = special_tokens
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def update_model_and_tokenizer(self):
        print("Old BART tokenizer length: " + str(len(self.tokenizer)))
        self.tokenizer.add_tokens(self.special_tokens, special_tokens=True)
        print("New BART tokenizer length: " + str(len(self.tokenizer)))
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model, self.tokenizer
    

if __name__ == "__main__":
    model_path = '/home/sda/wangzhijun/AllModels/bartt/BART-base'
    special_tokens = ['<entity_start>', '<entity_end>']
    
    updater = ModelUpdater(model_path, special_tokens)
    model, tokenizer = updater.update_model_and_tokenizer()

























# class ModelUpdater:
#     def __init__(self, model_path, special_tokens, descriptions):
#         self.model_path = model_path
#         self.special_tokens = special_tokens
#         self.descriptions = descriptions
#         self.model = BartForConditionalGeneration.from_pretrained(model_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#     def update_model_and_tokenizer(self):
#         print("Old BART tokenizer length: " + str(len(self.tokenizer)))
#         num_add_toks = self.tokenizer.add_tokens(self.special_tokens, special_tokens=True)
#         print("New BART tokenizer length: " + str(len(self.tokenizer)))
        
#         self.model.resize_token_embeddings(len(self.tokenizer))
        
#         with torch.no_grad():
#             for i, description in enumerate(self.descriptions):
#                 tokenized = self.tokenizer.tokenize(description)
#                 tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
#                 new_embedding = self.model.model.shared.weight[tokenized_ids].mean(axis=0)
#                 self.model.model.shared.weight[-num_add_toks + i, :] = new_embedding.clone().detach().requires_grad_(True)
        
#         return self.model, self.tokenizer
    
#     # Example usage in a main function
# if __name__ == "__main__":
#     model_path = '/home/sda/wangzhijun/AllModels/bartt/BART-base'
#     special_tokens = ['<start>', '<end>']
#     descriptions = ['start of entity', 'end of entity']
    
#     updater = ModelUpdater(model_path, special_tokens, descriptions)
#     model, tokenizer = updater.update_model_and_tokenizer()
    
#     # Verify the embeddings for the new tokens
#     for token in special_tokens:
#         token_id = tokenizer.convert_tokens_to_ids([token])[0]
#         print(f"Embedding for {token}: {model.model.shared.weight[token_id]}")