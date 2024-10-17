from transformers import AutoTokenizer
import torch


class Tokenizer():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def question_tokenizer(self, data):
        print("========== tokenizing questions ==========")
        self.questions = self.tokenizer(data['question'],
                                        truncation=True,
                                        max_length=self.tokenizer.model_max_length,
                                        stride=self.tokenizer.model_max_length//2,
                                        padding='max_length',
                                        return_tensors='pt')

        print(f"total {len(self.questions['input_ids'])} questions\n")
        
        return self.questions

    def context_tokenizer(self, data):
        assert self.questions, f"call question tokenizer first"

        self.data = data
        
        print("========== tokenizing contexts ==========")
        self.contexts = self.tokenizer(self.data['context'],
                                        truncation=True,
                                        max_length=self.tokenizer.model_max_length,
                                        stride=self.tokenizer.model_max_length//2,
                                        return_overflowing_tokens=True,
                                        return_offsets_mapping=True,
                                        padding="max_length",
                                        return_tensors='pt')
        
        indices, cnt = torch.unique(self.contexts['overflow_to_sample_mapping'], return_counts=True)

        new_questions = {'input_ids': torch.zeros(self.contexts['input_ids'].shape),
                         'attention_mask': torch.zeros(self.contexts['attention_mask'].shape),
                         'token_type_ids': torch.zeros(self.contexts['token_type_ids'].shape)}

        start = 0
        for idx, c in zip(indices, cnt):
            new_questions['input_ids'][start:start+c] = self.questions['input_ids'][idx]
            new_questions['attention_mask'][start:start+c] = self.questions['attention_mask'][idx]
            new_questions['token_type_ids'][start:start+c] = self.questions['token_type_ids'][idx]
            start += c

        self.questions = new_questions
        
        print(f"questions: {len(self.questions['input_ids'])}, answers: {len(self.contexts['input_ids'])}\n")

        return self.questions, self.contexts
    
    def corpus_tokenizer(self, data):
        tokenized_corpus = self.tokenizer(data,
                                          truncation=True,
                                          padding='max_length',
                                          return_tensors='pt')
        
        return tokenized_corpus
