# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    cross_entropy,
)


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):
    """
    Added settings for (imported from BaseTransformerArgs):
    - norm_type      -> 'rms_norm'  (default) or 'layer_norm'
    - attn_type      -> 'llama'    (default) or 'gpt'
    - pos_embed_type -> 'rope'     (default) or 'learned'
    - ffn_activation -> 'silu'     (default) or 'gelu'

    New additions:
    - dropout        -> Global dropout rate to be used across attention and feedforward
    - use_gpt_init   -> If True, apply GPT-style weight init; else keep LLaMA-style
    """

    seed: int = 42
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: Optional[int] = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        """
        We just call super() with the new config arguments. Also, optionally,
        for 'pos_embed_type=learned', you may want to create a learnable
        positional embedding here if you prefer to handle it externally from 
        BaseTransformer. For minimal changes, we preserve the logic in base class.
        """
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        # Token embeddings
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        # (NEW) Optional learned positional embedding for GPT:
        if args.pos_embed_type == "learned":
            self.pos_embeddings = torch.nn.Embedding(args.max_seqlen, args.dim)
        else:
            self.pos_embeddings = None

        # Norm on final hidden states before output
        #   (We keep the same approach as LLaMA, but you could also change to 
        #    do it GPT style if needed.)
        #   For GPT exactness, you might add a final LN also, but let's keep it simpler.
        if args.norm_type == "layernorm":
            self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.bias)
            # Add dtype specification to match model dtype
            self.norm = self.norm.to(dtype=torch.bfloat16 if args.model_dtype == "bf16" else torch.float32)
        else:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        
        # drop tokens
        self.drop = nn.Dropout(args.dropout)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        # ----------------
        # Token Embeddings
        # ----------------
        h = self.tok_embeddings(token_values)

        # (NEW) If we have a learned pos embedding, add it here:
        if self.pos_embeddings is not None:
            positions = torch.arange(seqlen, device=token_values.device)
            # Clamp position indices if they exceed self.pos_embeddings.num_embeddings
            if seqlen > self.pos_embeddings.num_embeddings:
                # You could log a warning here:
                # logger.warning(f"Sequence length {seqlen} exceeds pos_embed size; clamping to {self.pos_embeddings.num_embeddings}")
                positions = positions.clamp_max(self.pos_embeddings.num_embeddings - 1)
            pos_emb = self.pos_embeddings(positions).unsqueeze(0).expand(bsz, -1, -1)
            h = self.drop(h + pos_emb)

        # -------------
        # Causal Mask
        # -------------
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        # --------------
        # Transformer
        # --------------
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        # ---------
        # Output
        # ---------
        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        """
        If use_gpt_init == True, we do GPT-like initialization.
        Otherwise fall back to the original LLaMA style.
        """
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))

        # We reset embeddings. If you add a learned-pos-emb, you'd also reset here
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        # (NEW) Reset learned position embedding if present:
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(
                self.pos_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # If LN or RMSNorm, we do that in their own reset
        self.norm.reset_parameters()

        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self):
        if self.args.use_gpt_init:
            # GPT-style initialization for top-level modules
            def _gpt_like_weights(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    with torch.no_grad():
                        if module.bias is not None:
                            module.bias.data.zero_()
                        if module.weight is not None:
                            module.weight.data.fill_(1.0)
                elif isinstance(module, RMSNorm):
                    # Properly reset RMSNorm just like LLaMA
                    module.reset_parameters()
            
            # 1) Apply GPT init to all modules in self
            self.apply(_gpt_like_weights)

            # 2) Ensure rope embeddings (if any) are also reset
            if self.rope_embeddings is not None:
                self.rope_embeddings.reset_parameters()

            # 3) Also re-initialize each TransformerBlock
            #    if it has a custom init_weights method:
            for block in self.layers:
                if hasattr(block, "init_weights"):
                    block.init_weights()
        else:
            # Default LLaMA-style initialization
            super().init_weights()


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        if model_args.attn_type == "llama":
            attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
        else:
            # GPT style: typically n_kv_heads == n_heads, no grouping
            attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size