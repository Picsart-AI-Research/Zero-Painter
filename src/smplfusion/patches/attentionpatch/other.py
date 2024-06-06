# import xformers
# import xformers.ops

def forward(self, x, context=None, mask=None):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    b, _, _ = q.shape
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, t.shape[1], self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * self.heads, t.shape[1], self.dim_head)
        .contiguous(),
        (q, k, v),
    )

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

    if mask is not None:
        raise NotImplementedError
    out = (
        out.unsqueeze(0)
        .reshape(b, self.heads, out.shape[1], self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, out.shape[1], self.heads * self.dim_head)
    )
    return self.to_out(1 - out)