import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import waveverse_scene.hmg.models.pos_encoding as pos_encoding
from waveverse_scene.hmg.utils.motion_process import recover_from_ric


class Text2Motion_Transformer(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()
        self.path_feature = MLP(2, 256, embed_dim, 3)
        self.trans_base = CrossCondTransBase(
            num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate
        )
        self.trans_head = CrossCondTransHead(
            num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate
        )
        self.block_size = block_size
        self.sample_time = 51
        self.num_vq = num_vq

    def forward(self, idxs, clip_feature, paths, m_pos):
        # idx: [128, 50], clip_feature [128, 512]
        path_feature = self.path_feature(paths)  # [128, 64, 512]
        if len(idxs) != 0:
            mpos_feature = self.path_feature(m_pos)  # [128, 50, 512]
        else:
            mpos_feature = None
        feat = self.trans_base(idxs, clip_feature, path_feature, mpos_feature)
        logits = self.trans_head(feat)
        return logits

    def sample(self, clip_feature, paths, net, val_dataset, if_categorial=False):
        m_pos = None
        xs = None
        for k in range(self.sample_time):
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature, paths, m_pos)
            logits = logits[:, -1, :]
            # A zero-token motion cannot be decoded. Require one motion token.
            if k == 0:
                logits[:, self.num_vq] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)

            cur_pose = net.forward_decoder(xs)
            cur_pose = val_dataset.inv_transform_torch(cur_pose)
            pred_xyz = recover_from_ric(cur_pose.float(), 22)
            cur_pos = pred_xyz[:, -1:, 0, [0, 2]]
            if m_pos is None:
                m_pos = cur_pos
            else:
                m_pos = torch.cat((m_pos, cur_pos), dim=1)

            if k == self.sample_time - 1:
                return xs[:, :-1]
        return xs


class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(
            embed_dim, block_size, n_head, drop_out_rate
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ]
        )
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, clip_feature, paths_feature, mpos_feature):
        if len(idx) == 0:
            token_embeddings = torch.cat(
                [self.cond_emb(clip_feature).unsqueeze(1), paths_feature], dim=1
            )
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)

            stacked = torch.stack((token_embeddings, mpos_feature), dim=2)
            merged_feature = stacked.view(
                token_embeddings.shape[0], token_embeddings.shape[1] * 2, token_embeddings.shape[2]
            )

            token_embeddings = torch.cat(
                [self.cond_emb(clip_feature).unsqueeze(1), paths_feature, merged_feature], dim=1
            )
            # [128, 1, 512] [128, 64, 512] [128, 100, 512]

        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
