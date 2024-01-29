import torch
import torch.nn as nn
import math

class MultiTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 768
        self.att_drop_prob = 0.1
        self.state_drop_prob = 0.5
        self.num_heads = 12
        self.size_per_head = self.dim // self.num_heads # 64
        self.Wq = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wk = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.W = nn.Linear(self.num_heads * self.size_per_head, self.dim)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)
        self.att_drop = nn.Dropout(self.att_drop_prob)
        self.state_drop = nn.Dropout(self.state_drop_prob)

    def calc_mask_score(self, attention_mask):
        """
        input bxn
        output bxhxnxn
        """
        mask_score = torch.zeros(attention_mask.size(0), self.num_heads, attention_mask.size(1), attention_mask.size(1))
        mask_score = mask_score + attention_mask[:, None, None, :]
        mask_score = (1.0 - mask_score) * -10000.
        return mask_score

    def SelfAttention(self, x, attention_mask):
        """
        input x: bxnxd
        Q, K, V : bxnx(hxs) -> bxnxhxs -> bxhxnxs
        attention_mask: # bxn
        1 normal token
        0 masked token
        output bxnxd
        """
        new_size = x.size()[:-1] + (self.num_heads, self.size_per_head) # b, n, h, s
        Q = self.Wq(x).view(*new_size).permute(0, 2, 1, 3)
        K = self.Wk(x).view(*new_size).permute(0, 2, 1, 3)
        V = self.Wv(x).view(*new_size).permute(0, 2, 1, 3)
        attention_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.dim)
        # attention mask here
        attention_score = attention_score + self.calc_mask_score(attention_mask)
        attention_score = nn.Softmax(dim=3)(attention_score)
        attention_score = self.att_drop(attention_score)
        O = torch.matmul(attention_score, V)
        O = self.W(O.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), -1)) # bxnxd
        O = self.state_drop(O)
        O = self.lm(x + O)
        return O

    def FFN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output)
        output = self.lm_ffn(x + output)
        return output

    def forward(self, x, attention_mask):
        """
        input x: bxnxd
        output bxnxd
        """
        x = self.SelfAttention(x, attention_mask)
        x = self.FFN(x)
        return x

if __name__ == '__main__':
    layer = MultiTransformerLayer()
    x = torch.zeros(2, 256, 768)
    mask = torch.ones(2, 256)
    print(layer(x, mask))