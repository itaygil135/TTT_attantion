import torch
import torch.nn.functional as F

def dumb_generate(model, tokenizer, prompt, max_length=512, temperature=1.0):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            # top-k sampling
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token_id)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)