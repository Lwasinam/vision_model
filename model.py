
##Implementation of tranformer from scratch, this implememtation was inspired by Umar Jamir

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)


        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model) 


class PositionEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model:int, batch: int) -> None:
        super(PositionEncoding, self).__init__()
        # self.seq_len = seq_len
        # self.d_model = d_model
        # self.batch = batch
        self.dropout = nn.Dropout(p=0.1)
    
        ##initialize the positional encoding with zeros
        positional_encoding = torch.zeros(seq_len, d_model)
     
        ##first path of the equation is postion/scaling factor per dimesnsion
        postion  = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
        ## this calculates the scaling term per dimension (512)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # div_term = torch.pow(10,  torch.arange(0,self.d_model, 2).float() *-4/self.d_model)
      

        ## this calculates the sin values for even indices
        positional_encoding[:, 0::2] = torch.sin(postion * div_term) 

      
        ## this calculates the cos values for odd indices
        positional_encoding[:, 1::2] = torch.cos(postion * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):  
         x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
         return self.dropout(x)

## code from @jankrepl on github
class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size : int
        Size of the patch (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The emmbedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.n_patches+1, embed_dim)
        )
         # Adding CLS token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(
                x
            )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS tokens for the batch
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed  # Learnable pos embed -> (n_samples, n_patches_embed_dim) 
    
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, heads: int) -> None:
        super(MultiHeadAttention,self).__init__()
        self.head = heads
        self.head_dim = d_model // heads
        


        assert d_model % heads == 0, 'cannot divide d_model by heads'

        ## initialize the query, key and value weights 512*512
        self.query_weight = nn.Linear(d_model, d_model, bias=False)
        self.key_weight = nn.Linear(d_model, d_model,bias=False)
        self.value_weight = nn.Linear(d_model, d_model,bias=False)
        self.final_weight  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.1)

      
    def self_attention(self,query, key, value, mask,dropout):
        #splitting query, key and value into heads
                #this gives us a dimension of batch, num_heads, seq_len by 64. basically 1 sentence is converted to have 8 parts (heads)
        query = query.view(query.shape[0], query.shape[1],self.head,self.head_dim).transpose(2,1)
        key = key.view(key.shape[0], key.shape[1],self.head,self.head_dim).transpose(2,1)
        value = value.view(value.shape[0], value.shape[1],self.head,self.head_dim).transpose(2,1)
        
        attention = query @ key.transpose(3,2)
        attention = attention / math.sqrt(query.shape[-1])
        # print(f' attention shape {attention.shape}')
        # print(f' mask shape {mask.shape}')

        if mask is not None:
           attention = attention.masked_fill((attention != 0).int().to(device) == 0, -1e9)      
        attention = torch.softmax(attention, dim=-1)      
        if dropout is not None:
            attention = dropout(attention)
        attention_scores =  attention @ value    
       
        return attention_scores.transpose(2,1).contiguous().view(attention_scores.shape[0], -1, self.head_dim * self.head)
      
    def forward(self,query, key, value,mask):

        ## initialize the query, key and value matrices to give us seq_len by 512
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)

        attention = MultiHeadAttention.self_attention(self, query, key, value, mask, self.dropout)
        return self.final_weight(attention) 

class FeedForward(nn.Module):
    def __init__(self,d_model:int, d_ff:int ) -> None:
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)  # Fully connected layer 1
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Fully connected layer 2
     
    
    def forward(self,x ):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))  

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) :
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)   

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, head:int, d_ff:int) -> None:
        super(EncoderBlock, self).__init__()    
        self.multiheadattention = MultiHeadAttention(d_model,head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x, src_mask):
       # Self-attention block
        norm = self.layer_norm1(x)
        attention = self.multiheadattention(norm, norm, norm, src_mask)
        x = (x + self.dropout1(attention))

        # Feedforward block
        norm2 = self.layer_norm2(x)
        ff = self.feedforward(x)
        return x + self.dropout2(ff)     

class Encoder(nn.Module):
    def __init__(self, number_of_block:int, d_model:int, head:int, d_ff:int) -> None:
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # Use nn.ModuleList to store the EncoderBlock instances
        self.encoders = nn.ModuleList([EncoderBlock(d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask):
        for encoder_block in self.encoders:
            x = encoder_block(x, src_mask)
        return self.norm(x)   
   
class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, head:int, d_ff:int) -> None:
        super(DecoderBlock, self).__init__()
        self.head_dim = d_model // head
        
        self.multiheadattention = MultiHeadAttention(d_model, head)
        self.crossattention = MultiHeadAttention(d_model, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model,d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    def forward(self, x, src_mask, tgt_mask, encoder_output):
        #Self-attention block
        norm = self.layer_norm1(x)
        attention = self.multiheadattention(norm, norm, norm, tgt_mask)
        x = (x + self.dropout1(attention))
    
        # Cross-attention block
        norm2 = self.layer_norm2(x)    
        cross_attention = self.crossattention(norm, encoder_output, encoder_output, src_mask)
        x = (x + self.dropout2(cross_attention))
   
        # Feedforward block  
        norm3  = self.layer_norm3(x)
        ff = self.feedforward(norm3)
        return x + self.dropout3(ff)   


class Decoder(nn.Module):
    def __init__(self, number_of_block:int,d_model:int, head:int, d_ff:int) -> None:
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(d_model) 
        self.decoders = nn.ModuleList([DecoderBlock(d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask, tgt_mask, encoder_output):
        for decoder_block in self.decoders:
            x = decoder_block(x, src_mask, tgt_mask, encoder_output)
        return self.norm(x)    


class Transformer(nn.Module):
    def __init__(self, seq_len:int, batch:int, d_model:int,target_vocab_size:int, head: int = 8, d_ff: int =  2048, number_of_block: int = 4, imgSize: int = 224, patch_size: int = 14) -> None:
        super(Transformer, self).__init__()
    
       
        self.encoder = Encoder(number_of_block,d_model, head, d_ff )
        self.decoder = Decoder(number_of_block, d_model, head, d_ff )
        self.patch_embeddings = PatchEmbed(imgSize, patch_size)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.projection = ProjectionLayer(d_model, target_vocab_size)
        self.target_embedding = InputEmbeddings(d_model,target_vocab_size)
        self.positional_encoding = PositionEncoding(seq_len, d_model, batch)

   
    def encode(self,x, src_mask):
        x  = self.patch_embeddings(x)
        # x = self.source_embedding(x)
        # x = self.positional_encoding(x)
        return self.encoder(x, src_mask)
       
    def decode(self,x, src_mask, tgt_mask, encoder_output):
        x = self.target_embedding(x)
        x = self.positional_encoding(x)
        return self.decoder(x, src_mask, tgt_mask, encoder_output,)
        
    def project(self, x):
        return self.projection(x)
        


def build_transformer(seq_len, batch, target_vocab_size,  d_model)-> Transformer:
    

    transformer = Transformer(seq_len, batch,  d_model, target_vocab_size )

      #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer         


# import torch
# import torch.nn as nn
# import math

# class LayerNormalization(nn.Module):

#     def __init__(self, eps:float=10**-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
#         self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_size)
#          # Keep the dimension for broadcasting
#         mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # Keep the dimension for broadcasting
#         std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # eps is to prevent dividing by zero or when std is very small
#         return self.alpha * (x - mean) / (std + self.eps) + self.bias

# class FeedForwardBlock(nn.Module):

#     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
#         super().__init__()
#         self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

#     def forward(self, x):
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
#         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# class InputEmbeddings(nn.Module):

#     def __init__(self, d_model: int, vocab_size: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         # (batch, seq_len) --> (batch, seq_len, d_model)
#         # Multiply by sqrt(d_model) to scale the embeddings according to the paper
#         return self.embedding(x) * math.sqrt(self.d_model)
    
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len
#         self.dropout = nn.Dropout(dropout)
#         # Create a matrix of shape (seq_len, d_model)
#         pe = torch.zeros(seq_len, d_model)
#         # Create a vector of shape (seq_len)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
#         # Create a vector of shape (d_model)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
#         # Apply sine to even indices
#         pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
#         # Apply cosine to odd indices
#         pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
#         # Add a batch dimension to the positional encoding
#         pe = pe.unsqueeze(0) # (1, seq_len, d_model)
#         # Register the positional encoding as a buffer
#         pe = pe.transpose(1,2)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
#         return self.dropout(x)

# class ResidualConnection(nn.Module):
    
#         def __init__(self, dropout: float) -> None:
#             super().__init__()
#             self.dropout = nn.Dropout(dropout)
#             self.norm = LayerNormalization()
    
#         def forward(self, x, sublayer):
#             return x + self.dropout(sublayer(self.norm(x)))

# class MultiHeadAttentionBlock(nn.Module):

#     def __init__(self, d_model: int, h: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model # Embedding vector size
#         self.h = h # Number of heads
#         # Make sure d_model is divisible by h
#         assert d_model % h == 0, "d_model is not divisible by h"

#         self.d_k = d_model // h # Dimension of vector seen by each head
#         self.w_q = nn.Linear(d_model, d_model) # Wq
#         self.w_k = nn.Linear(d_model, d_model) # Wk
#         self.w_v = nn.Linear(d_model, d_model) # Wv
#         self.w_o = nn.Linear(d_model, d_model) # Wo
#         self.dropout = nn.Dropout(dropout)

#     @staticmethod
#     def attention(query, key, value, mask, dropout: nn.Dropout):
#         d_k = query.shape[-1]
#         # Just apply the formula from the paper
#         # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
       
#         attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
       
       
#         if mask is not None:
#             # Write a very low value (indicating -inf) to the positions where mask == 0
#             attention_scores.masked_fill_(mask == 0, -1e9)
#         attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
#         if dropout is not None:
#             attention_scores = dropout(attention_scores)
#         # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
#         # return attention scores which can be used for visualization
#         return (attention_scores @ value), attention_scores

#     def forward(self, q, k, v, mask):
#         query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

#         # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
#         query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
#         key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
#         value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

#         # Calculate attention
#         x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        
#         # Combine all the heads together
#         # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

#         # Multiply by Wo
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
#         return self.w_o(x)

# # class EncoderBlock(nn.Module):

# #     def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
# #         super().__init__()
# #         self.self_attention_block = self_attention_block
# #         self.feed_forward_block = feed_forward_block
# #         self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

# #     def forward(self, x, src_mask):
# #         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
# #         x = self.residual_connections[1](x, self.feed_forward_block)
# #         return x
    
# # class Encoder(nn.Module):

# #     def __init__(self, layers: nn.ModuleList) -> None:
# #         super().__init__()
# #         self.layers = layers
# #         self.norm = LayerNormalization()

# #     def forward(self, x, mask):
# #         for layer in self.layers:
# #             x = layer(x, mask)
# #         return self.norm(x)
# class EncoderBlock(nn.Module):
#     def __init__(self, d_model:int, head:int, d_ff:int) -> None:
#         super(EncoderBlock, self).__init__()    
#         self.multiheadattention = MultiHeadAttentionBlock(d_model,head, 0.1)
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(p=0.1)
#         self.feedforward = FeedForwardBlock(d_model, d_ff, 0.1)
#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.layer_norm3 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(p=0.1)

#     def forward(self, x, src_mask):
#        # Self-attention block
#         norm = self.layer_norm1(x)
#         attention = self.multiheadattention(norm, norm, norm, src_mask)
#         x = (x + self.dropout1(attention))

#         # Feedforward block
#         norm2 = self.layer_norm2(x)
#         ff = self.feedforward(x)
#         return x + self.dropout2(ff)    

# class Encoder(nn.Module):
#     def __init__(self, number_of_block:int, d_model:int, head:int, d_ff:int) -> None:
#         super(Encoder, self).__init__()
#         self.norm = nn.LayerNorm(d_model)
        
#         # Use nn.ModuleList to store the EncoderBlock instances
#         self.encoders = nn.ModuleList([EncoderBlock(d_model, head, d_ff) 
#                                        for _ in range(number_of_block)])

#     def forward(self, x, src_mask):
#         for encoder_block in self.encoders:
#             x = encoder_block(x, src_mask)
#         return self.norm(x)  

# class ProjectionLayer(nn.Module):

#     def __init__(self, d_model, vocab_size) -> None:
#         super().__init__()
#         self.proj = nn.Linear(d_model, vocab_size)

#     def forward(self, x) -> None:
#         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
#         return torch.log_softmax(self.proj(x), dim = -1)

# class DecoderBlock(nn.Module):
#     def __init__(self, d_model:int, head:int, d_ff:int) -> None:
#         super(DecoderBlock, self).__init__()
#         self.head_dim = d_model // head
        
#         self.multiheadattention = MultiHeadAttentionBlock(d_model, head, 0.1)
#         self.crossattention = MultiHeadAttentionBlock(d_model, head, 0.1)
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(p=0.1)
#         self.feedforward = FeedForwardBlock(d_model,d_ff, 0.1)
#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.layer_norm3 = nn.LayerNorm(d_model)
#         self.layer_norm4 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(p=0.1)
#         self.dropout3 = nn.Dropout(p=0.1)
#     def forward(self, x, src_mask, tgt_mask, encoder_output):
#          # Self-attention block
#         norm = self.layer_norm1(x)
#         attention = self.multiheadattention(norm, norm, norm, tgt_mask)
#         x = (x + self.dropout1(attention))
    
#         # Cross-attention block
#         norm2 = self.layer_norm2(x)    
#         cross_attention = self.crossattention(norm, encoder_output, encoder_output, src_mask)
#         x = (x + self.dropout2(cross_attention))
   
#         # Feedforward block  
#         norm3  = self.layer_norm3(x)
#         ff = self.feedforward(norm3)
#         return x + self.dropout3(ff)  


# class Decoder(nn.Module):
#     def __init__(self, number_of_block:int,d_model:int, head:int, d_ff:int) -> None:
#         super(Decoder, self).__init__()
#         self.norm = nn.LayerNorm(d_model) 
#         self.decoders = nn.ModuleList([DecoderBlock(d_model, head, d_ff) 
#                                        for _ in range(number_of_block)])

#     def forward(self, x, src_mask, tgt_mask, encoder_output):
#         for decoder_block in self.decoders:
#             x = decoder_block(x, src_mask, tgt_mask, encoder_output)
#         return self.norm(x)          
    


# class Transformer(nn.Module):
#     def __init__(self, seq_len:int, batch:int, d_model:int,target_vocab_size:int, source_vocab_size:int, head: int = 8, d_ff: int =  2048, number_of_block: int = 6, dropout: float = 0.1) -> None:
#         super(Transformer, self).__init__()
    
       
#         self.encoder = Encoder(number_of_block,d_model, head, d_ff )
#         self.decoder = Decoder(number_of_block, d_model, head, d_ff )
        

#         # encoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
#         # feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         # self.encoder = Encoder(nn.ModuleList([EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout) for _ in range(number_of_block)]))


#         # decoder_self_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
#         # decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, head, dropout)
#         # feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         # self.decoder = Decoder(nn.ModuleList([DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout) for _ in range(number_of_block) ]))

#         self.projection = ProjectionLayer(d_model, target_vocab_size)
#         self.source_embedding = InputEmbeddings(d_model,source_vocab_size )
#         self.target_embedding = InputEmbeddings(d_model,target_vocab_size)
#         self.positional_encoding = PositionalEncoding(seq_len, d_model, dropout)

   
#     def encode(self,x, src_mask):
#         x = self.source_embedding(x)
#         x = self.positional_encoding(x)
#         return self.encoder(x, src_mask)
       
#     def decode(self,encoder_output, src_mask, x,  tgt_mask):
#         x = self.target_embedding(x)
#         x = self.positional_encoding(x)
#         return self.decoder(x, src_mask, tgt_mask, encoder_output)
        
#     def project(self, x):
#         return self.projection(x)
        


# def build_transformer(seq_len, batch, target_vocab_size, source_vocab_size,  d_model)-> Transformer:
    

#     transformer = Transformer(seq_len, batch,  d_model,  target_vocab_size, source_vocab_size )

#       #Initialize the parameters
#     for p in transformer.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return transformer    