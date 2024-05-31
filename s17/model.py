import torch
import torch.nn as nn
import math
from config_file import get_config

config = get_config()


class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10e-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * ( x - mean) / (std + self.eps) + self.bias
 

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # squeeze and expand approach
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        # d_model is model's dimension, raw sequences will be transformed via this
        # here `d_model` is equal to all concatenated heads' dimension 
        # NOTE: MODEL CAPACITY controlled by d_model
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # creating embeddings here --> random values
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        # here we multiply with sqrt(d_model) to scale the embeddings (as per the paper)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        # The positional encodings that are used to add position information to the input embeddings

        self.d_model = d_model
        self.seq_len  = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        
        # normal arange like numpy but unsqueezing at dim=1 results in tensor 
        # of shape [seq_len, 1] ([[0], [1]...[seq_len]])
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # log and exp for numerical stability 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0/d_model)))
        
        # positional encodings
        # even starting from 0, step by 2 --> sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # odd starting from 1, step by 2 -> cos
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # to know about Pytorch's register_buffer : https://stackoverflow.com/a/57541778/3903762
        self.register_buffer('pe', pe) # stored as part of model, moved to appropriate device but not trained

    def forward(self, x):
        # (batch_size, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout()
        self.norm = LayerNormalization()
    
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model:int,
                 h:int,
                 dropout:float
                 ):
        super().__init__()

        assert d_model % h == 0, "MAN!!! your `d_model` is not perfectly divisible by `h`"
        self.h = h
        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias=False) # combined query matrix
        self.w_k = nn.Linear(d_model, d_model, bias=False) # combined Key matrix
        self.w_v = nn.Linear(d_model, d_model, bias=False) # value
        self.w_o = nn.Linear(d_model, d_model, bias=False) # final projection matrix to be used after merging all heads
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask , dropout:nn.Dropout):

        #mask can be either encoder or decoder mask
        d_k = query.shape[-1] # get last dim from incoming query for attention calculations

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # d_k refers to in-use dim or part of d_model

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # where mask is 0 -> fill with small neg value
        attention_scores = attention_scores.softmax(dim=-1) 
        # (batch, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # ( batch_size, h, seq_len) --> (batch_size, h, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) #   (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) #   (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) #   (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        #   (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attenstion_scores = MultiheadAttentionBlock.attention(query=query,
                                                                      key=key,
                                                                      value=value,
                                                                      mask=mask,
                                                                      dropout=self.dropout)
        
        # Now combine all heads together
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) --> (batch, -1 (remaining seq_len), h * d_k (it was d_model))
        # reverse of previous break-down into heads step here
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # NOTE: call to contiguous here

        # now use Out matrix
        return self.w_o(x)
    

class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiheadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):

        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiheadAttentionBlock,
                 cross_attention_block: MultiheadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # this is self-attention in Decoder, the first attention block in architecture diagram
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # this is the cross attention block, getting Q from itself while K and V comes from encoded output from Encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Feed Forward block as the third block
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # this is the final projection layer on top of decoder and `vocab_size` is output vocab size
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEmbedding,
                 tgt_pos: PositionalEmbedding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)  # convert source seq to embedding
        src = self.src_pos(src)  # convert embedded source sequence to have positional embeddings encoded within it

        return self.encoder(src, src_mask)

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)  # embed the target sequences
        tgt = self.tgt_pos(tgt)  # add positional encoding to target embedded sequences
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # final projection layer with output vocab size
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # create positional encoding layers next
    src_pos = PositionalEmbedding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEmbedding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)

    # create list to hold Encoder blocks
    encoder_blocks = []
    # create a list to hold Decoder blocks
    decoder_blocks = []

    # In case `parameter sharing` is enabled, reduce block number by 2
    if config['param_sharing'] is True:
        N = N // 2

    for _ in range(N):  # repeat for N
        # create or initialize sub-layers for current Encoder block
        encoder_self_attention = MultiheadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # create 1 instance of Encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)

        # add this 1 block to Encoder block list
        encoder_blocks.append(encoder_block)

    for _ in range(N):  # have `N` total blocks
        # initialize sub-layers for each block
        decoder_self_attention_block = MultiheadAttentionBlock(d_model=d_model, h=h, dropout=dropout)

        decoder_cross_attention_block = MultiheadAttentionBlock(d_model=d_model, h=h, dropout=dropout)

        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # create 1 decoder block now
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block,
                                     cross_attention_block=decoder_cross_attention_block,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)

        # add this 1 decoder block to decoder blocks list
        decoder_blocks.append(decoder_block)

    if config['param_sharing'] is True:
        e1, e2, e3 = encoder_blocks
        d1, d2, d3 = decoder_blocks
        encoder_blocks = [e1, e2, e3, e3, e2, e1]
        decoder_blocks = [d1, d2, d3, d3, d2, d1]

    # link encoders
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))

    # link decoders
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # create a projection layer for decoder side
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    # create the Transformer
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed, tgt_embed=tgt_embed,
                              src_pos=src_pos, tgt_pos=tgt_pos,
                              projection_layer=projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer