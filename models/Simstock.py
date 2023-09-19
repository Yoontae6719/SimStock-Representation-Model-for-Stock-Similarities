import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import wraps
from math import sqrt
from einops import rearrange, repeat
import copy
from layers.Embed import Sector_embedding


def loss_fn(x, y):
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


# helper
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        
        
# Notice that we donot tune our model.


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

    
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases
    
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class model(nn.Module):
    def __init__(self, args, device):
        super(model, self).__init__()
        
        self.lambda_values =  args.lambda_values
        
        self.init_lin_h = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_lin_c = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_input = nn.Linear(args.noise_dim, args.latent_dim)

        self.rnn = nn.LSTM(args.latent_dim, args.latent_dim, args.num_rnn_layer)
        
        self.NumericalEmbedder = NumericalEmbedder(args.sector_emb,args.latent_dim)
        
        # Transforming LSTM output to vector shape
        self.lin_transform_down = nn.Sequential(
                            nn.Linear(args.latent_dim, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, (args.data_size+1)*256 + 256*256 + 256 + 256+ 1))
        
        # Transforming vector to LSTM input shape
        self.lin_transform_up = nn.Sequential(
                            nn.Linear( (args.data_size+1)*256 + 256*256 + 256 + 256+ 1, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, args.latent_dim))
        
        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = args.data_size
        self.device = device
        
        # Sector embedding
        self.sector_emb = Sector_embedding(args.sector_emb, args.sector_size)
        self.sector_projection = nn.Linear(args.sector_emb, args.noise_dim) # input size
        
        # Projection term
        
        self.online_encoder = nn.Linear(args.noise_dim, args.noise_dim)
        
        self.target_encoder = None
        
        ### Attn params
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.sector_emb))
        self.scale = (256 // 4) ** -0.5
        self.dropout = nn.Dropout(0.05)
        self.depth = 2
        
        self.mlp = FeedForward(256) #FeedForward(256)
        self.norm = nn.LayerNorm(256)
    
    def nn_construction_attns(self, E):
        # First layers
        attn_qw  = E[:, :(self.data_size + 1)* 256 ] # dim * heads
        attn_kw  = E[:, :(self.data_size + 1)* 256 ] 
        attn_vw  = E[:, :(self.data_size + 1)* 256 ]
        
        attn_out_w = E[:, (self.data_size + 1)* 256 : (self.data_size + 1)* 256 + 256*256 ]
        attn_out_b = E[:, (self.data_size + 1)* 256 + 256*256 : (self.data_size + 1)* 256 + 256*256 + 256]

        out_m = E[:, (self.data_size + 1)* 256 + 256*256 + 256 : (self.data_size + 1)* 256 + 256*256 + 256 +  256]
        out_b = E[:, -1]

        return  [attn_qw.view((self.data_size +1), 256), 
                 attn_kw.view((self.data_size +1), 256), 
                 attn_vw.view((self.data_size +1), 256),
                 
                 attn_out_w.view(-1, 256),
                 attn_out_b.view(-1, 256),
                 
                 out_m.view(-1, 1),
                 out_b
                ] 
    
    def forward(self, X, z, sector, E=None, hidden=None, return_embedding = False):
        s = self.sector_emb(sector.squeeze(1))
        s = self.sector_projection(s)
        
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))
                
        #### add cls toekn ####
        X = torch.add(X, s)
        X_out = X # Anchor param
        
        X = self.NumericalEmbedder(X)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = X.shape[0])
        X = torch.cat((cls_tokens, X), dim = 1) # X shape : torch.Size([256, 26, 256])
        
        
        ############ Attn (1-depth)####################
        m_list = self.nn_construction_attns(E)        
        
        if return_embedding == True:

            q = torch.einsum('bnd,nd->bnd', X, m_list[0]) # attn_q
            k = torch.einsum('bnd,nd->bnd', X, m_list[1]) # attn_k
            v = torch.einsum('bnd,nd->bnd', X, m_list[2]) # attn_v

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 4), (q, k, v))
            q = q * self.scale
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
            attn = sim.softmax(dim = -1)
            dropped_attn = self.dropout(attn)

            out = torch.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)', h = 4)


            out = torch.einsum('b f i, i j-> b f j', out, m_list[3]) + m_list[4]
            output = out + X

            output = torch.einsum('b f i, i j-> b f j', output, m_list[5]) + m_list[6]
            output = output[:, 0]
            return output, attn
        #############################################
        
        positive, negative = self.augment(X, self.lambda_values)
        # f_\theta (weight sharing)

        pos_q = torch.einsum('bnd,nd->bnd', positive, m_list[0]) # attn_q
        pos_k = torch.einsum('bnd,nd->bnd', positive, m_list[1]) # attn_k
        pos_v = torch.einsum('bnd,nd->bnd', positive, m_list[2]) # attn_v
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 4), (pos_q, pos_k, pos_v))
        pos_q = pos_q * self.scale
        pos_sim = torch.einsum('b h i d, b h j d -> b h i j', pos_q, pos_k)
        pos_attn = pos_sim.softmax(dim = -1)
        pos_dropped_attn = self.dropout(pos_attn)
        pos_out = torch.einsum('b h i j, b h j d -> b h i d', pos_dropped_attn, pos_v)
        pos_out = rearrange(pos_out, 'b h n d -> b n (h d)', h = 4)
        pos_out = torch.einsum('b f i, i j-> b f j', pos_out, m_list[3]) + m_list[4]
        pos_output = pos_out + positive
        pos_output = torch.einsum('b f i, i j-> b f j', pos_output, m_list[5]) + m_list[6]
        pos_output = pos_output[:, 0]
        
        neg_q = torch.einsum('bnd,nd->bnd', negative, m_list[0]) # attn_q
        neg_k = torch.einsum('bnd,nd->bnd', negative, m_list[1]) # attn_k
        neg_v = torch.einsum('bnd,nd->bnd', negative, m_list[2]) # attn_v
        neg_q, neg_k, neg_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 4), (neg_q, neg_k, neg_v))
        neg_q = neg_q * self.scale
        neg_sim = torch.einsum('b h i d, b h j d -> b h i j', neg_q, neg_k)
        neg_attn = neg_sim.softmax(dim = -1)
        neg_dropped_attn = self.dropout(neg_attn)
        neg_out = torch.einsum('b h i j, b h j d -> b h i d', neg_dropped_attn, neg_v)
        neg_out = rearrange(neg_out, 'b h n d -> b n (h d)', h = 4)
        neg_out = torch.einsum('b f i, i j-> b f j', neg_out, m_list[3]) + m_list[4]
        neg_output = neg_out + negative
        neg_output = torch.einsum('b f i, i j-> b f j', neg_output, m_list[5]) + m_list[6]
        neg_output = neg_output[:, 0]

        loss = triplet_loss(X_out, pos_output, neg_output)
        
        
        return E, hidden, loss
    
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder    
    
    def augment(self, x, lambda_value, use_cuda=True):
        batch_size, token, feature_dim = x.shape
        if use_cuda:
            index1 = torch.randperm(feature_dim).cuda()
            index2 = torch.randperm(feature_dim).cuda()
        else:
            index1 = torch.randperm(feature_dim)
            index2 = torch.randperm(feature_dim)

        mixed_x1 = lambda_value * x + (1 - lambda_value) * x[:,  :, index1]
        mixed_x2 = (1 - lambda_value) * x + lambda_value * x[:,  :, index2]

        return mixed_x1, mixed_x2
