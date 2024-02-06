import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import ViTFeatureExtractor

# import model
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(
    model_id
)

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
    
     

        
        src_target_pair = self.ds[idx]
        src_image = src_target_pair['image']
        tgt_text = src_target_pair['en_text']
        
        enc_input = feature_extractor(
        src_image,
        return_tensors='pt'
        )
        
        
     
        ## image will be resized to be fed to pathc emnedding as a tansor -> (1, 3, 224, 224)


        # # Transform the text into tokens
        # enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # # Add <s> and </s> token
        # encoder_input = torch.cat(
        #     [
        #         self.sos_token,
        #         torch.tensor(enc_input_tokens, dtype=torch.int64),
        #         self.eos_token,
        #         torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        #     ],
        #     dim=0,
        # )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        # assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {
            'encoder_input': enc_input['pixel_values'][0],
            'decoder_input': decoder_input,
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
             
            # "src_text": src_text,
            "tgt_text": tgt_text,
        }

   
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
        