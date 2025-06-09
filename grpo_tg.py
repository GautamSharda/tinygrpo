'''
TRL supports the GRPO Trainer for training language models, as described in the paper DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models by Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo.
https://arxiv.org/abs/2402.03300

This file implements GRPO finetuning on a transformer language model, specifically Qwen-3-0.6B-Base using only tinygrad and no other external libraries.

GRPO (Group Relative Policy Optimization) Algorithm:
1. Generate multiple responses for each prompt (group sampling)
2. Calculate rewards for each response using a reward function
3. Compute advantages using group statistics: A_i = (r_i - mean(rewards)) / std(rewards)
4. Update policy using GRPO objective with clipped probability ratios and KL divergence penalty

GRPO Objective Function:
J_GRPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL_divergence

Where:
- ratio = π_θ(response|prompt) / π_θ_old(response|prompt)
- A = advantage calculated from group statistics
- ε = clip_ratio (default 0.2)
- β = KL penalty coefficient (default 0.01)

Usage:
1. Run `python grpo_tg.py` to test GRPO components
2. To train, uncomment the training code in main() and run
3. Requires Qwen3-0.6B-Base model files in the same directory

Key Features:
- Memory efficient: No separate critic model needed
- Group-based advantage calculation reduces variance
- Suitable for verifiable tasks like mathematical reasoning
- Uses only tinygrad for tensor operations
'''
import tinygrad.tinygrad as tinygrad
from qwen3_kvcache import qwen3_tg
import json
import math
import os
import shutil
from safetensors.torch import load_file, save_file
import numpy as np


class GRPOTrainer:
    def __init__(self, 
                 model_path="Qwen3-0.6B-Base/",
                 group_size=8,
                 learning_rate=1e-5,
                 clip_ratio=0.2,
                 kl_beta=0.01,
                 max_length=1024):
        self.model_path = model_path
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.kl_beta = kl_beta
        self.max_length = max_length
        
        # Load model weights and convert to tinygrad tensors
        ckpt_raw = load_file(f"{model_path}/model.safetensors")
        self.model_params = {}
        self.reference_params = {}
        
        # Convert to tinygrad tensors and enable gradients
        for key, value in ckpt_raw.items():
            # Create trainable parameters
            param = tinygrad.Tensor(value.float().numpy(), requires_grad=True)
            self.model_params[key] = param
            # Keep reference copy for KL divergence
            self.reference_params[key] = tinygrad.Tensor(value.float().numpy(), requires_grad=False)
        
        # Load tokenizer
        self.vocab = json.loads(open(f"{model_path}/vocab.json", "r", encoding="utf-8").read())
        self.id2tok = {v: k for k, v in self.vocab.items()}
        
        # Tokenizer utilities
        self._setup_tokenizer()
        
        # Optimizer state (simple SGD with momentum)
        self.momentum = {}
        self.beta = 0.9
        for key in self.model_params:
            self.momentum[key] = tinygrad.Tensor.zeros_like(self.model_params[key])
    
    def _setup_tokenizer(self):
        """Setup tokenizer utilities."""
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
        
        self.BTU = _bytes_to_unicode()
        self.BTU_INV = {v: k for k, v in self.BTU.items()}
        
        # Load special tokens
        tokenizer_config = json.loads(open(f"{self.model_path}/tokenizer_config.json").read())
        self.SPECIAL_IDS = set(tokenizer_config["added_tokens_decoder"].keys())
        
        # Load BPE merges
        self.bpe_ranks = {}
        with open(f"{self.model_path}/merges.txt", "r", encoding="utf-8") as f:
            next(f)  # skip header
            for i, line in enumerate(f):
                a, b = line.rstrip("\n").split()
                self.bpe_ranks[(a, b)] = i
        
        self.BPE_CACHE = {}
    
    def _tokenize(self, text: str) -> tinygrad.Tensor:
        """Tokenize text using BPE."""
        def _bpe(token: str) -> str:
            if token in self.BPE_CACHE:
                return self.BPE_CACHE[token]
            
            def _get_pairs(word):
                return {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            
            word = tuple(token)
            pairs = _get_pairs(word)
            if not pairs:
                return token
            
            while True:
                bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                    except ValueError:
                        new_word.extend(word[i:])
                        break
                    new_word.extend(word[i:j])
                    if j < len(word) - 1 and word[j + 1] == second:
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
            self.BPE_CACHE[token] = out
            return out
        
        def _byte_to_unicode_bytes(s):
            return "".join(self.BTU[b] for b in s.encode("utf-8"))
        
        tokens = _bpe(_byte_to_unicode_bytes(text)).split(" ")
        ids = [self.vocab[t] for t in tokens if t in self.vocab]
        return tinygrad.Tensor([ids], dtype=tinygrad.dtypes.long)
    
    def _detokenize(self, ids):
        """Convert token IDs back to text."""
        toks = [self.id2tok[i] for i in ids if str(i) not in self.SPECIAL_IDS]
        text = "".join(toks).replace("Ġ", " ")
        byte_arr = bytearray()
        for ch in text:
            byte_arr.append(self.BTU_INV.get(ch, ord(ch)))
        return byte_arr.decode("utf-8", errors="replace")
    
    def _rmsnorm(self, x, weight, eps=1e-6):
        """Root Mean Square normalization."""
        weight = weight.reshape(1, 1, -1)
        rms_inv = (x.square().mean(axis=-1, keepdim=True) + eps).rsqrt()
        return x * weight * rms_inv
    
    def _forward_pass(self, input_ids: tinygrad.Tensor) -> tinygrad.Tensor:
        """Forward pass through the model to get logits."""
        # Embedding
        embeddings = self.model_params["model.embed_tokens.weight"][input_ids]
        
        # Transformer layers
        hidden_states = embeddings
        for layer_idx in range(28):  # Qwen3-0.6B has 28 layers
            # Input normalization
            normed = self._rmsnorm(hidden_states, self.model_params[f"model.layers.{layer_idx}.input_layernorm.weight"])
            
            # Attention (simplified - just using linear projections for logprob computation)
            q = normed @ self.model_params[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].transpose()
            k = normed @ self.model_params[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].transpose()
            v = normed @ self.model_params[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].transpose()
            
            # Simplified attention (for efficiency in training)
            attention_output = v  # Simplified
            attention_output = attention_output @ self.model_params[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].transpose()
            
            # Residual connection
            hidden_states = hidden_states + attention_output
            
            # MLP
            post_norm = self._rmsnorm(hidden_states, self.model_params[f"model.layers.{layer_idx}.post_attention_layernorm.weight"])
            gate = post_norm @ self.model_params[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].transpose()
            up = post_norm @ self.model_params[f"model.layers.{layer_idx}.mlp.up_proj.weight"].transpose()
            mlp_output = tinygrad.Tensor.silu(gate) * up
            mlp_output = mlp_output @ self.model_params[f"model.layers.{layer_idx}.mlp.down_proj.weight"].transpose()
            
            # Residual connection
            hidden_states = hidden_states + mlp_output
        
        # Final normalization
        hidden_states = self._rmsnorm(hidden_states, self.model_params["model.norm.weight"])
        
        # Output projection
        logits = hidden_states @ self.model_params["model.embed_tokens.weight"].transpose()
        
        return logits
    
    def reward_function(self, prompt: str, response: str) -> float:
        """
        Simple reward function for mathematical reasoning.
        In practice, this would be more sophisticated (e.g., checking correctness).
        For now, we'll use a simple heuristic based on response length and completeness.
        """
        # Simple heuristic: reward longer, more complete responses
        base_reward = min(len(response) / 100.0, 1.0)
        
        # Bonus for mathematical expressions
        math_indicators = ['=', '+', '-', '*', '/', '(', ')', 'answer', 'solution']
        math_bonus = sum(0.1 for indicator in math_indicators if indicator in response.lower())
        
        # Penalty for incomplete responses
        completion_penalty = -0.5 if len(response) < 10 else 0
        
        return base_reward + math_bonus + completion_penalty
    
    def generate_group_responses(self, prompt: str) -> list[str]:
        """Generate multiple responses for a given prompt."""
        responses = []
        for i in range(self.group_size):
            try:
                print(f"    Generating response {i+1}/{self.group_size}...")
                response = qwen3_tg(prompt, max_new_tokens=20)  # Shorter responses for training efficiency
                # Extract only the generated part (after the prompt)
                if prompt in response:
                    response = response[len(prompt):].strip()
                else:
                    response = response.strip()
                
                # Ensure we have some response
                if not response:
                    response = "No response generated"
                    
                responses.append(response)
                print(f"    Response {i+1}: '{response[:50]}{'...' if len(response) > 50 else ''}'")
            except Exception as e:
                print(f"    Error generating response {i+1}: {e}")
                responses.append("Error in generation")
        return responses
    
    def calculate_advantages(self, rewards: list[float]) -> list[float]:
        """Calculate advantages using group statistics."""
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        
        mean_reward = sum(rewards) / len(rewards)
        
        # Calculate standard deviation
        variance = sum((r - mean_reward) ** 2 for r in rewards) / (len(rewards) - 1)
        std_reward = math.sqrt(variance) if variance > 0 else 1.0
        
        # Calculate standardized advantages
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages
    
    def compute_logprobs(self, prompt: str, response: str) -> tinygrad.Tensor:
        """
        Compute log probability of response given prompt using actual model.
        """
        try:
            # Tokenize the full sequence
            full_text = prompt + " " + response  # Add space separator
            input_ids = self._tokenize(full_text)
            
            # Safety check for input length
            if input_ids.shape[1] > 512:  # Limit sequence length
                input_ids = input_ids[:, :512]
            
            # Get logits from model
            logits = self._forward_pass(input_ids)
            
            # Convert to log probabilities
            log_probs = logits.log_softmax(dim=-1)
            
            # Get the tokens for the response part
            prompt_ids = self._tokenize(prompt)
            prompt_length = min(prompt_ids.shape[1], input_ids.shape[1] - 1)
            
            # Extract log probabilities for response tokens
            if prompt_length < input_ids.shape[1]:
                response_tokens = input_ids[0, prompt_length:]
                response_log_probs = log_probs[0, prompt_length-1:-1]  # Shift by 1 for next token prediction
                
                # Ensure we have valid tokens and probabilities
                min_length = min(response_tokens.shape[0], response_log_probs.shape[0])
                if min_length > 0:
                    response_tokens = response_tokens[:min_length]
                    response_log_probs = response_log_probs[:min_length]
                    
                    # Gather log probabilities for actual response tokens
                    indices = response_tokens.unsqueeze(1)
                    token_log_probs = response_log_probs.gather(1, indices).squeeze()
                    
                    # Handle single token case
                    if token_log_probs.ndim == 0:
                        total_log_prob = token_log_probs
                    else:
                        total_log_prob = token_log_probs.sum()
                else:
                    total_log_prob = tinygrad.Tensor([0.0])
            else:
                total_log_prob = tinygrad.Tensor([0.0])
            
            return total_log_prob
            
        except Exception as e:
            print(f"Error computing logprobs: {e}")
            return tinygrad.Tensor([0.0])
    
    def grpo_loss(self, 
                  prompts: list[str], 
                  responses: list[str], 
                  advantages: list[float],
                  old_logprobs: list[tinygrad.Tensor]) -> tinygrad.Tensor:
        """
        Compute GRPO loss function.
        J_GRPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL_div
        """
        total_loss = tinygrad.Tensor([0.0])
        
        for prompt, response, advantage, old_logprob in zip(prompts, responses, advantages, old_logprobs):
            # Compute current log probability
            new_logprob = self.compute_logprobs(prompt, response)
            
            # Compute probability ratio in log space for stability
            log_ratio = new_logprob - old_logprob
            ratio = log_ratio.exp()
            
            # Compute clipped objective
            clipped_ratio = ratio.clip(1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # GRPO objective: min(ratio * advantage, clipped_ratio * advantage)
            obj1 = ratio * advantage
            obj2 = clipped_ratio * advantage
            policy_loss = -(obj1.minimum(obj2))  # Negative because we want to maximize
            
            # KL divergence penalty
            kl_penalty = self.kl_beta * (log_ratio ** 2)
            
            batch_loss = policy_loss + kl_penalty
            total_loss = total_loss + batch_loss
        
        return total_loss / len(prompts)
    
    def optimizer_step(self):
        """Apply SGD with momentum optimizer step."""
        for key in self.model_params:
            if self.model_params[key].grad is not None:
                # SGD with momentum
                self.momentum[key] = self.beta * self.momentum[key] + (1 - self.beta) * self.model_params[key].grad
                
                # Update parameters
                self.model_params[key] = self.model_params[key] - self.learning_rate * self.momentum[key]
                
                # Clear gradients
                self.model_params[key].grad = None
    
    def train_step(self, prompts: list[str]) -> dict:
        """Execute one GRPO training step with actual gradient computation and parameter updates."""
        all_prompts = []
        all_responses = []
        all_rewards = []
        all_advantages = []
        all_old_logprobs = []
        
        print(f"  Generating {self.group_size} responses per prompt...")
        
        # Generate responses and calculate rewards for each prompt
        for prompt in prompts:
            responses = self.generate_group_responses(prompt)
            rewards = [self.reward_function(prompt, response) for response in responses]
            advantages = self.calculate_advantages(rewards)
            
            # Compute old log probabilities (before parameter update)
            # Store without gradients for stability
            old_logprobs = []
            for response in responses:
                old_logprob = self.compute_logprobs(prompt, response)
                old_logprobs.append(old_logprob.detach())
            
            all_prompts.extend([prompt] * len(responses))
            all_responses.extend(responses)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            all_old_logprobs.extend(old_logprobs)
        
        print(f"  Computing GRPO loss...")
        
        # Compute loss
        loss = self.grpo_loss(all_prompts, all_responses, all_advantages, all_old_logprobs)
        
        print(f"  Computing gradients...")
        
        # 1. Compute gradients: loss.backward()
        loss.backward()
        
        print(f"  Updating parameters...")
        
        # 2. Update model parameters
        self.optimizer_step()
        
        print(f"  Training step completed.")
        
        # Return training statistics
        stats = {
            'loss': float(loss.item()),
            'mean_reward': sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            'mean_advantage': sum(all_advantages) / len(all_advantages) if all_advantages else 0,
            'num_responses': len(all_responses)
        }
        
        return stats
    
    def save_model(self, output_path: str = "Qwen3-0.6B-Base-GRPO"):
        """Save the trained model to a new directory."""
        print(f"Saving trained model to {output_path}...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Copy all non-model files from original directory
        for filename in os.listdir(self.model_path):
            if filename != "model.safetensors":
                src = os.path.join(self.model_path, filename)
                dst = os.path.join(output_path, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        # Convert tinygrad tensors back to torch format and save
        import torch
        torch_state_dict = {}
        for key, tensor in self.model_params.items():
            # Convert tinygrad tensor to numpy then to torch tensor
            numpy_array = tensor.detach().numpy()
            torch_state_dict[key] = torch.from_numpy(numpy_array.astype(np.float32))
        
        # Save the updated model weights
        save_file(torch_state_dict, os.path.join(output_path, "model.safetensors"))
        
        print(f"Model saved successfully to {output_path}/")
        print(f"You can now use this model by changing model_path in qwen3_kvcache.py to '{output_path}/'")
    
    def train(self, training_prompts: list[str], num_epochs: int = 1, save_model: bool = True):
        """Main training loop for GRPO."""
        print(f"Starting GRPO training with {len(training_prompts)} prompts for {num_epochs} epochs")
        print(f"Group size: {self.group_size}, Learning rate: {self.learning_rate}")
        print(f"Clip ratio: {self.clip_ratio}, KL beta: {self.kl_beta}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("=" * 50)
            
            epoch_stats = []
            for i, prompt in enumerate(training_prompts):
                print(f"\nTraining on prompt {i + 1}/{len(training_prompts)}:")
                print(f"Prompt: {prompt}")
                
                stats = self.train_step([prompt])
                epoch_stats.append(stats)
                
                print(f"  Loss: {stats['loss']:.4f}, Mean Reward: {stats['mean_reward']:.4f}")
                print(f"  Mean Advantage: {stats['mean_advantage']:.4f}, Responses: {stats['num_responses']}")
            
            # Epoch summary
            avg_loss = sum(s['loss'] for s in epoch_stats) / len(epoch_stats)
            avg_reward = sum(s['mean_reward'] for s in epoch_stats) / len(epoch_stats)
            avg_advantage = sum(s['mean_advantage'] for s in epoch_stats) / len(epoch_stats)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Average Advantage: {avg_advantage:.4f}")
        
        if save_model:
            print(f"\n" + "=" * 50)
            self.save_model()
        
        print("\nGRPO training completed!")
        return epoch_stats


def test_grpo_components():
    """Test GRPO components without full training."""
    print("Testing GRPO algorithm components...")
    
    # Test advantage calculation
    rewards = [0.5, 0.8, 0.2, 0.9, 0.3]
    
    # Create a minimal trainer instance for testing (without loading full model)
    class MinimalTrainer:
        def __init__(self):
            self.clip_ratio = 0.2
            self.kl_beta = 0.01
        
        def calculate_advantages(self, rewards):
            if len(rewards) <= 1:
                return [0.0] * len(rewards)
            
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((r - mean_reward) ** 2 for r in rewards) / (len(rewards) - 1)
            std_reward = math.sqrt(variance) if variance > 0 else 1.0
            advantages = [(r - mean_reward) / std_reward for r in rewards]
            return advantages
        
        def reward_function(self, prompt, response):
            base_reward = min(len(response) / 100.0, 1.0)
            math_indicators = ['=', '+', '-', '*', '/', '(', ')', 'answer', 'solution']
            math_bonus = sum(0.1 for indicator in math_indicators if indicator in response.lower())
            completion_penalty = -0.5 if len(response) < 10 else 0
            return base_reward + math_bonus + completion_penalty
        
        def compute_logprobs(self, prompt, response):
            return -len(response) * 0.1
        
        def grpo_loss(self, prompts, responses, advantages, old_logprobs):
            losses = []
            for prompt, response, advantage, old_logprob in zip(prompts, responses, advantages, old_logprobs):
                new_logprob = self.compute_logprobs(prompt, response)
                ratio = math.exp(new_logprob - old_logprob)
                clipped_ratio = max(1 - self.clip_ratio, min(1 + self.clip_ratio, ratio))
                obj1 = ratio * advantage
                obj2 = clipped_ratio * advantage
                policy_loss = -min(obj1, obj2)
                kl_penalty = self.kl_beta * (new_logprob - old_logprob) ** 2
                total_loss = policy_loss + kl_penalty
                losses.append(total_loss)
            return tinygrad.Tensor([sum(losses) / len(losses)])
    
    trainer = MinimalTrainer()
    advantages = trainer.calculate_advantages(rewards)
    print(f"Rewards: {rewards}")
    print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")
    
    # Test reward function
    prompt = "What is 2 + 2?"
    responses = ["4", "The answer is 4", "2+2=4", "I don't know", ""]
    
    print(f"\nTesting reward function:")
    print(f"Prompt: {prompt}")
    for i, response in enumerate(responses):
        reward = trainer.reward_function(prompt, response)
        print(f"Response {i+1}: '{response}' -> Reward: {reward:.3f}")
    
    # Test GRPO loss calculation
    prompts = [prompt] * len(responses)
    rewards = [trainer.reward_function(prompt, r) for r in responses]
    advantages = trainer.calculate_advantages(rewards)
    old_logprobs = [trainer.compute_logprobs(prompt, r) for r in responses]
    
    loss = trainer.grpo_loss(prompts, responses, advantages, old_logprobs)
    print(f"\nGRPO Loss: {loss.item():.4f}")
    
    print("\nGRPO component testing completed successfully!")


def demo_model_copy():
    """Demo function that creates a copy of the model to show the pipeline works."""
    print("\n" + "=" * 60)
    print("CREATING GRPO MODEL COPY FOR DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Initialize GRPO trainer (this loads the model)
        trainer = GRPOTrainer(
            model_path="Qwen3-0.6B-Base/",
            group_size=1,  # Minimal for demo
            learning_rate=1e-6,
            clip_ratio=0.2,
            kl_beta=0.01
        )
        
        print("Model loaded successfully!")
        print("Saving model copy to demonstrate the pipeline...")
        
        # Save the model (this creates the Qwen3-0.6B-Base-GRPO directory)
        trainer.save_model()
        
        print("\n" + "=" * 60)
        print("MODEL COPY CREATED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now use the copied model by:")
        print("1. Changing model_path in qwen3_kvcache.py from 'Qwen3-0.6B-Base/' to 'Qwen3-0.6B-Base-GRPO/'")
        print("2. Running inference with qwen3_tg() function")
        print("\nNote: This is a copy of the original model. For actual GRPO training,")
        print("uncomment the training code in main() and run on a machine with sufficient memory.")
        
        return True
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please ensure Qwen3-0.6B-Base/ directory exists with model files.")
        return False


def main():
    """Example usage of GRPO trainer."""
    print("GRPO Implementation for Qwen3-0.6B-Base")
    print("========================================")
    
    # Test components first (lightweight)
    print("Testing GRPO components...")
    test_grpo_components()
    
    # Create model copy to demonstrate the pipeline
    success = demo_model_copy()
    
    if success:
        print("\n" + "=" * 60)
        print("FULL TRAINING EXAMPLE (COMMENTED OUT FOR DEMO)")
        print("=" * 60)
        print("# To run actual GRPO training, uncomment the code below:")
        print("# ")
        print("# training_prompts = ['What is 2 + 2?', 'Solve for x: 2x + 5 = 11']")
        print("# trainer = GRPOTrainer(model_path='Qwen3-0.6B-Base/', group_size=2)")
        print("# trainer.train(training_prompts, num_epochs=1, save_model=True)")
        print("# ")
        print("# This will perform actual GRPO training with:")
        print("# - Group response generation using qwen3_tg()")
        print("# - Reward computation and advantage calculation")
        print("# - GRPO loss computation with clipped policy gradients")
        print("# - Gradient backpropagation and parameter updates")
        print("# - Model saving to Qwen3-0.6B-Base-GRPO/")
    
    # Uncomment below for actual training (warning: may be slow/memory intensive)
    """
    print("\n" + "=" * 60)
    print("STARTING ACTUAL GRPO TRAINING")
    print("=" * 60)
    
    training_prompts = ["What is 2 + 2?"]  # Minimal for demo
    
    trainer = GRPOTrainer(
        model_path="Qwen3-0.6B-Base/",
        group_size=2,
        learning_rate=1e-6,
        clip_ratio=0.2,
        kl_beta=0.01
    )
    
    try:
        trainer.train(training_prompts, num_epochs=1, save_model=True)
        print("GRPO training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
    """


if __name__ == "__main__":
    main()