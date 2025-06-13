"""
qwen3.py implements Qwen3-0.6B-Base in tinygrad with text generation. 
Some implemented transformer functions include: byte-pair encoding, rotary positional embedding, group query attention, & root-mean-squared normalization.
Generate text with qwen3_tg().
"""
def test(prompt: str, max_new_tokens=5):
    import time
    tg_times = []
    hf_times = []
    tg_out = None
    hf_out = None
    for _ in range(1):
        start_time = time.time()
        tg_out = qwen3_tg(prompt, max_new_tokens)
        tg_times.append(time.time() - start_time)
        start_time = time.time()
        hf_out = qwen3_hf(prompt, max_new_tokens)
        hf_times.append(time.time() - start_time)
    avg_tg_time = sum(tg_times) / len(tg_times)
    avg_hf_time = sum(hf_times) / len(hf_times)
    tg_tokens_per_sec = max_new_tokens / avg_tg_time
    hf_tokens_per_sec = max_new_tokens / avg_hf_time

    print("TG:")
    print(tg_out, f"[{tg_tokens_per_sec:.2f} Toks/S]")
    print("\nHF:")
    print(hf_out, f"[{hf_tokens_per_sec:.2f} Toks/S]")
    print("\nVerdict:", tg_out == hf_out)


def qwen3_tg(prompt: str, max_new_tokens=5):
    import tinygrad.tinygrad as tinygrad
    from safetensors.torch import load_file # Change with tinygrad torch_load in nn.state.py
    import json
    ckpt = load_file("Qwen3-0.6B-Base/model.safetensors") # Load only the weights you need as you need them in functions, don't pass this around
    VOCAB = json.loads(open("Qwen3-0.6B-Base/vocab.json", "r", encoding="utf-8").read())
    ID2TOK = {v: k for k, v in VOCAB.items()}
    def _bytes_to_unicode():
        bs = list(range(33, 127)) + list(range(161, 172)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, map(chr, cs)))
    BTU = _bytes_to_unicode()
    BTU_INV = {v: k for k, v in BTU.items()}
    SPECIAL_IDS = set(json.loads(
        open("Qwen3-0.6B-Base/tokenizer_config.json").read()
    )["added_tokens_decoder"].keys())
    def _tokenize(text: str,
            *,
            add_bos: bool = False,
            add_eos: bool = False,
            device: str = tinygrad.Device.DEFAULT,
            max_length: int | None = None) -> tinygrad.Tensor:
        """
        Byte‑level BPE encode Qwen‑style.
        Returns shape (1, seq_len) int64 tensor on `device`.
        """    
        # `merges.txt` – first line is a header you can ignore
        bpe_ranks = {}
        with open("Qwen3-0.6B-Base/merges.txt", "r", encoding="utf‑8") as f:
            next(f)                               # skip header
            for i, line in enumerate(f):
                a, b = line.rstrip("\n").split()
                bpe_ranks[(a, b)] = i
        BPE_CACHE = {}
        # Special‑token list (ids are in `tokenizer_config.json`)
        SPECIAL = {}
        added = json.loads(open("Qwen3-0.6B-Base/tokenizer_config.json", "r", encoding="utf-8").read())["added_tokens_decoder"]
        for t in added.keys():
            SPECIAL[added[t]["content"]] = int(t)
        BOS = SPECIAL.get("<|im_start|>")
        EOS = SPECIAL.get("<|im_end|>")     
        def _bpe(token: str) -> str:
            """Byte‑pair‑encode one token.  Returns space‑separated merged symbols."""
            def _get_pairs(word):
                """Return set of symbol pairs in a word (tuple of symbols)."""
                return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

            if token in BPE_CACHE:
                return BPE_CACHE[token]
            word = tuple(token)
            pairs = _get_pairs(word)
            if not pairs:
                return token                         # no merges possible
            while True:
                # pick the lowest‑ranked bigram present in `word`
                bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
                if bigram not in bpe_ranks:          # none left to merge
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                    except ValueError:               # `first` not found
                        new_word.extend(word[i:])
                        break
                    new_word.extend(word[i:j])       # stuff before `first`
                    # always append first symbol
                    if j < len(word) - 1 and word[j + 1] == second:
                        # merge the bigram
                        new_word.append(first + second)
                        i = j + 2
                    else:
                        new_word.append(word[j])
                        i = j + 1
                word = tuple(new_word)
                if len(word) == 1:
                    break
                pairs = _get_pairs(word)
            out = " ".join(word)
            BPE_CACHE[token] = out
            return out
        def _byte_to_unicode_bytes(s):
            return "".join(BTU[b] for b in s.encode("utf‑8"))

        tokens = _bpe(_byte_to_unicode_bytes(text)).split(" ")
        ids = [VOCAB[t] for t in tokens if t in VOCAB]
        if add_bos and BOS is not None:
            ids.insert(0, BOS)
        if add_eos and EOS is not None:
            ids.append(EOS)
        if max_length is not None:
            ids = ids[:max_length]
        return tinygrad.Tensor([ids], dtype=tinygrad.dtypes.long, device=device)
    def _detokenize(ids, _=None):
        """
        Inverse of `_tokenize`:
        list[int]  →  human‑readable UTF‑8 string.
        ‑ skips special‑token IDs
        ‑ restores spaces from the 'Ġ' marker
        ‑ converts GPT‑2 byte‑runes back to raw bytes, then decodes UTF‑8
        """
        toks = [ID2TOK[i] for i in ids if str(i) not in SPECIAL_IDS]
        text = "".join(toks).replace("Ġ", " ")
        byte_arr = bytearray()
        for ch in text:
            byte_arr.append(BTU_INV.get(ch, ord(ch)))
        return byte_arr.decode("utf-8", errors="replace")
    def _embed(input_ids: tinygrad.Tensor,
            *,
            weights: tinygrad.Tensor) -> tinygrad.Tensor:
        """
        Args
        ----
        input_ids : (B, L) int64
        weight    : (V, d) float/bfloat

        Returns
        -------
        (B, L, d) tensor with the same dtype/device as `weight`.
        """
        input_ids = input_ids.to(weights.device)
        return weights[input_ids]
    def _rmsnorm(activations: tinygrad.Tensor, gamma: tinygrad.Tensor, epsilon: float = 1e-6):
        gamma = gamma.reshape(1, 1, -1)
        rms_inv = (activations.square().mean(axis=-1, keepdim=True) + epsilon).rsqrt()
        return activations * gamma * rms_inv
    def _transformer(E: tinygrad.Tensor, i) -> tinygrad.Tensor:
        def _qk_norm(Q, K, gamma_q, gamma_k, eps=1e-6):
            # Q : (B,L,16,128)   K : (B,L,8,128)
            rms_q = (Q.square().mean(axis=-1, keepdim=True) + eps).rsqrt()
            rms_k = (K.square().mean(axis=-1, keepdim=True) + eps).rsqrt()
            gq = gamma_q.reshape(1,1,1,-1)   # (1,1,1,128)
            gk = gamma_k.reshape(1,1,1,-1)
            Q = Q * rms_q * gq
            K = K * rms_k * gk
            return Q, K
        def _apply_rope(Q: tinygrad.Tensor,
                    K: tinygrad.Tensor) -> tuple[tinygrad.Tensor, tinygrad.Tensor]:
            """
            Match HuggingFace's exact RoPE implementation - no complex rotation, just duplication
            """
            def _build_sincos(max_len=32768, d_head=128, theta=1000000):
                # Exact HuggingFace formula converted to TinyGrad
                dim = d_head  # 128
                base = theta  # 100000.0
                # torch.arange(0, dim, 2) -> tinygrad.Tensor.arange(0, dim, 2)
                indices = tinygrad.Tensor.arange(0, dim, 2).float()  # [0, 2, 4, ..., 126] (64 elements)
                inv_freq = 1.0 / (base ** (indices / dim))           # (64,)
                pos = tinygrad.Tensor.arange(max_len).float().reshape(-1, 1)     # (L, 1)
                freqs = pos @ inv_freq.reshape(1, -1)                            # (L, 64) matrix multiply
                # Duplicate frequencies: (L, 64) -> (L, 128)
                freqs_full = freqs.cat(freqs, dim=-1)                            # (L, 128)
                return {"sin": freqs_full.sin(), "cos": freqs_full.cos()}        # both (L, 128)
            def _rotate_half(x):
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                result = (-x2).cat(x1, dim=-1)
                return result

            sincos = _build_sincos()
            B, H, L, d = Q.shape
            cos = sincos["cos"][:L].reshape(1, L, 128)  # Reshape to match HF
            sin = sincos["sin"][:L].reshape(1, L, 128)
            cos = cos.unsqueeze(1)  # (1, L, 128) → (1, 1, L, 128)
            sin = sin.unsqueeze(1)  # (1, L, 128) → (1, 1, L, 128)
            q_cos = Q * cos
            q_rot_sin = _rotate_half(Q) * sin
            q_embed = q_cos + q_rot_sin
            k_embed = (K * cos) + (_rotate_half(K) * sin)
            return q_embed, k_embed
        def _flash_gqa(Q, K, V, mask: tinygrad.Tensor | None = None):
            B, Hq, L, d = Q.shape           # (1, 16, 7, 128)
            Hk = K.shape[1]                 # 8
            n_rep = Hq // Hk                # 2
            # K: (1, 8, 7, 128) -> (1, 16, 7, 128)
            K = K.unsqueeze(2).expand(B, Hk, n_rep, L, d).reshape(B, Hk * n_rep, L, d)
            V = V.unsqueeze(2).expand(B, Hk, n_rep, L, d).reshape(B, Hk * n_rep, L, d)
            context = Q.scaled_dot_product_attention(
                K, V, 
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=True
            )
            context = context.transpose(1, 2)  # (B, H, L, d) -> (B, L, H, d)
            return context

        h = _rmsnorm(E, gamma=tinygrad.Tensor(ckpt[f"model.layers.{i}.input_layernorm.weight"].float().numpy()))
        # 2.  linear projections -----------------------------------------------
        Q = h @ tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.q_proj.weight"].float().numpy()).transpose()       # (B,L,2048)
        K = h @ tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.k_proj.weight"].float().numpy()).transpose()       # (B,L,1024)
        V = h @ tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.v_proj.weight"].float().numpy()).transpose()       # (B,L,1024)
        # 3.  reshape into heads -----------------------------------------------
        Q = Q.reshape(*Q.shape[:-1], 16, 128)       # (B,L,16,128)
        K = K.reshape(*K.shape[:-1],  8, 128)       # (B,L,8,128)
        V = V.reshape(*V.shape[:-1],  8, 128)       # (B,L,8,128)
        # 4. QK normalization ---------------------------------------------------
        Q, K = _qk_norm(Q, K, 
                        gamma_q=tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.q_norm.weight"].float().numpy()),
                        gamma_k=tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.k_norm.weight"].float().numpy()))
        # 5. Transpose to match HuggingFace layout -----------------------------
        Q = Q.transpose(1, 2)  # (B,L,16,128) → (B,16,L,128)  
        K = K.transpose(1, 2)  # (B,L,8,128) → (B,8,L,128)
        V = V.transpose(1, 2)  # (B,L,8,128) → (B,8,L,128)
        # 7. Apply RoPE --------------------------------------------------------
        Q, K = _apply_rope(Q, K)
        # 8.  Grouped‑Query Attention (stub) ------------------------------------
        att = _flash_gqa(Q, K, V)    # (B,L,16,128)
        # 9.  concat heads & output proj ---------------------------------------
        att = att.reshape(*E.shape[:-1], 2048)           # (B,L,1024)
        o = att @ tinygrad.Tensor(ckpt[f"model.layers.{i}.self_attn.o_proj.weight"].float().numpy()).transpose()                  # residual + Wo
        x = o + E
        pn = _rmsnorm(x, tinygrad.Tensor(ckpt[f"model.layers.{i}.post_attention_layernorm.weight"].float().numpy()))
        up   = pn @ tinygrad.Tensor(ckpt[f"model.layers.{i}.mlp.up_proj.weight"].float().numpy()).transpose()                    # (B,L,3072)
        gate = pn @ tinygrad.Tensor(ckpt[f"model.layers.{i}.mlp.gate_proj.weight"].float().numpy()).transpose()                  # (B,L,3072)
        ffn  = tinygrad.Tensor.silu(gate) * up
        ffn  = ffn @ tinygrad.Tensor(ckpt[f"model.layers.{i}.mlp.down_proj.weight"].float().numpy()).transpose()                # (B,L,1024)
        act = ffn + x                           # residual
        return act

    generation = prompt
    # Embed the initial prompt once
    input_tokens = _tokenize(generation)
    generation_ids = input_tokens.tolist()[0]  # Start with initial tokens
    embeddings = _embed(input_tokens, weights=tinygrad.Tensor(ckpt["model.embed_tokens.weight"].float().numpy()))

    for _ in range(max_new_tokens):
        # Use existing embeddings instead of re-embedding everything
        act = embeddings
        for l in range(28):
            act = _transformer(act, l)
        # -------- 10. final RMSNorm + logits ---------------------------------
        final_gamma = tinygrad.Tensor(ckpt["model.norm.weight"].float().numpy())
        last_hidden = _rmsnorm(act, final_gamma)            # (1, L, 1024)
        logits = last_hidden @ tinygrad.Tensor(              # (1, L, V)
            ckpt["model.embed_tokens.weight"].float().numpy()
        ).transpose()
        # -------- 11. greedy sampling ---------------------------------------
        next_id = int(logits[:, -1].argmax(-1).item())       # scalar python int
        # Add the new token to our running list
        generation_ids.append(next_id)
        # Only embed the new token and concatenate
        new_token = tinygrad.Tensor([[next_id]], dtype=tinygrad.dtypes.long)
        new_embedding = _embed(new_token, weights=tinygrad.Tensor(ckpt["model.embed_tokens.weight"].float().numpy()))
        embeddings = embeddings.cat(new_embedding, dim=1)  # Concatenate along sequence dimension
        # stop when <|im_end|> is reached
        if next_id == 151645:
            break

    # -------- 12. decode the full sequence once at the end ----------------
    generation = _detokenize(generation_ids, None)
    return generation


def qwen3_hf(prompt: str, max_new_tokens=5):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")  
    # Add this near the top of the file, after the imports
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, # None
        pad_token_id=tokenizer.pad_token_id,  # <|endoftext|> (Optional)
        max_new_tokens=max_new_tokens
    )
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


def compare_matrices(matrix_file_one_path: str, matrix_file_two_path: str):
    import numpy as np
    matrix_one = np.loadtxt(matrix_file_one_path, delimiter=',')
    matrix_two = np.loadtxt(matrix_file_two_path, delimiter=',')
    print("Differences between:", matrix_file_one_path, matrix_file_two_path)    
    diff_array = matrix_one - matrix_two
    print(f"Max absolute difference: {np.max(np.abs(diff_array))}")
    print(f"Mean absolute difference: {np.mean(np.abs(diff_array))}")
    print(f"RMS difference: {np.sqrt(np.mean(diff_array**2))}")
    if np.allclose(matrix_one, matrix_two, rtol=1e-3, atol=1e-5):
        print("Matrices are numerically equivalent (within normal floating point tolerance)")
    else:
        print("Matrices have significant differences")


if __name__ == "__main__":
    prompt: str = "Hi, how are you doing?"
    test(prompt)
    # import time
    # start_time = time.time()
    # try:
    #     # qwen3_hf(prompt)
    #     qwen3_tg(prompt)
    # except Exception as e:
    #     elapsed_time = time.time() - start_time
    #     print(f"Exception occurred after {elapsed_time:.2f} seconds: {e}")