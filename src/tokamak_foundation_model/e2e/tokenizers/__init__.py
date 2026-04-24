"""Per-modality tokenizers.

Each tokenizer maps a raw 50 ms signal window for one modality to a sequence
of tokens shaped ``(batch, n_tokens, d_model)`` with an added modality
embedding and positional encoding. All tokenizer weights are trained
end-to-end with the backbone (``ResearchPlan.MD`` §3.3).
"""