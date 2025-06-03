import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowTimeToTokenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.group_size = 154  # 每组总token数
        self.num_query_tokens = 10  # 每组中Q token数
        self.num_kv_tokens = 144  # 每组中KV token数

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [seq_len, embed_dim]
        Returns:
            输出张量 [seq_len, embed_dim]
        """
        seq_len, embed_dim = x.shape
        output = x.clone()

        num_groups = seq_len // self.group_size
        
        # 生成所有索引 (向量化操作)
        group_starts = torch.arange(0, num_groups, device=x.device) * self.group_size
        
        # 所有Q的索引 (每组前10个)
        all_q_indices = (group_starts.unsqueeze(1) + torch.arange(self.num_query_tokens, device=x.device)).view(-1)
        
        # 所有KV的索引 (每组后144个)
        all_kv_indices = (group_starts.unsqueeze(1) + torch.arange(self.num_query_tokens, self.group_size, device=x.device)).view(-1)
        
        # group_boundaries (每个组的KV数量固定为144)
        group_boundaries = torch.arange(0, num_groups + 1, device=x.device) * self.num_kv_tokens
        
        # 提取并投影所有Q
        all_q = x[all_q_indices, :]  # [total_q, D]
        q = self.q_proj(all_q)       # [total_q, D]
        
        # 提取并投影所有KV
        all_kv = x[all_kv_indices, :]  # [total_kv, D]
        k = self.k_proj(all_kv)        # [total_kv, D]
        v = self.v_proj(all_kv)
        
        # 重塑为多头形式
        q = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, total_q, D]
        k = k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, total_kv, D]
        v = v.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, total_kv, D]
        
        # 计算注意力分数 (KV对Q的注意力)
        attn_scores = torch.matmul(k, q.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [H, total_kv, total_q]
        
        # 创建mask确保每个KV只关注对应组的Q
        # mask = torch.zeros_like(attn_scores)
        # for i in range(1, len(group_boundaries)):
        #     start = group_boundaries[i-1]
        #     end = group_boundaries[i]
        #     mask[:, start:end, (i-1)*self.num_query_tokens:i*self.num_query_tokens] = 1
        # 向量化生成mask
        total_kv = all_kv.size(0)
        total_q = all_q.size(0)

        k_indices = torch.arange(total_kv, device=x.device)
        g_kv = k_indices // self.num_kv_tokens
        start_q = g_kv * self.num_query_tokens

        start_q = start_q.view(-1, 1)
        end_q = start_q + self.num_query_tokens
        q_indices = torch.arange(total_q, device=x.device).view(1, -1)

        mask_2d = (q_indices >= start_q) & (q_indices < end_q)
        mask_2d = mask_2d.to(dtype=attn_scores.dtype)

        mask = mask_2d.unsqueeze(0).expand(self.num_heads, -1, -1)

        
        attn_scores = attn_scores.masked_fill(mask == 0, -6e4)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力 (用Q的值更新KV)
        group_output = torch.matmul(attn_probs, q)

        group_output = group_output.transpose(0, 1).reshape(-1, embed_dim)  # [total_kv, D]
        group_output = self.out_proj(group_output)
        
        output[all_kv_indices, :] = group_output
        
        return output