from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
import matplotlib.pyplot as plt

import wandb
from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, None)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).long().to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).long().to(device)
      

        # calculate output
        out =model.decode(decoder_input,None,  decoder_mask, encoder_output)
     

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
     
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).long().fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, global_step,num_examples=3):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, None, tokenizer_tgt, max_len, device)

            # source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            # print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # if writer:
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('graph.png')
def batch_iterator(data):
    for i in range(0, len(data)):
        yield data[i]['en_text'] 

# Assuming batch_iterator is a function that yields batches
def tqdm_batch_iterator(data, *args, **kwargs):
    for batch in tqdm(batch_iterator(data, *args, **kwargs), total=len(data)):
        yield batch               

def get_or_build_tokenizer(config, ds,):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(tqdm_batch_iterator(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(path="youssef101/artelingo", name='artelingo', splits=['val','test'])
    # ds_raw = concatenate_datasets([ds_raw['val'], ds_raw['test']])
    ds_raw = load_dataset("HausaNLP/HausaVG", split='train+validation+test+challenge_test')
    # print(ds_raw[0])
  
    # Build tokenizers
    # tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw)
    seed = 20  # You can choose any integer as your seed
    torch.manual_seed(seed)
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_tgt,  config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,  tokenizer_tgt, config['seq_len'])

    # # Find the maximum length of each sentence in the source and target sentence
    # max_len_src = 0
    # max_len_tgt = 0

    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
    #     tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max length of source sentence: {max_len_src}')
    # print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_tgt

def get_model(config,  vocab_tgt_len):
    model = build_transformer( config['seq_len'],config['batch_size'], vocab_tgt_len, config['d_model'] )
    return model

def train_model(config):
    wandb.login(key = 'c20a1022142595d7d1324fdc53b3ccb34c0ded22')
    wandb.init(project="Vision", name=config['project_name'])

    # Initialize WandB configuration
    wandb.config.epochs = config['num_epochs']
    wandb.config.batch_size = config['batch_size']
    wandb.config.learning_rate = config['lr'] 
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, eps=1e-9)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95) 

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
 

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input= batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
           
            encoder_output = model.encode(encoder_input, None) # (B, seq_len, d_model)
            decoder_output = model.decode( decoder_input,None,  decoder_mask, encoder_output) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)
           
             # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
       
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            wandb.log({"Training Loss": loss.item(), "Global Step": global_step})
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            plot_grad_flow(model.named_parameters())

            # Update the weights
            optimizer.step()
            # scheduler.step()
            

            global_step += 1
        model.eval()
        eval_loss = 0.0
        # batch_iterator = tqdm(v_dataloader, desc=f"Processing Epoch {epoch:02d}")
        with torch.no_grad():
            for batch in val_dataloader:
            

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
            
                encoder_output = model.encode(encoder_input, None) # (B, seq_len, d_model)
                decoder_output = model.decode( decoder_input,None,  decoder_mask, encoder_output) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output)
            
                # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
        
                eval_loss += loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
           
                
        avg_val_loss = eval_loss / len(val_dataloader)
        print(f'Epoch {epoch},Validation Loss: {avg_val_loss.item()}')
        wandb.log({"Validation Loss": avg_val_loss.item(), "Global Step": global_step})

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

     


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
