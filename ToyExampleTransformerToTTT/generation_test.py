""" printing model generations"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import model

if __name__ == "__main__":
    # defining model
    model_instance = model.Transformer.load_model('checkpoints/model.pt')

    # importing gpt2 tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # tokenizing prompt
    prompt = "The future of AI in healthcare is"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # generating text
    generated_ids = model_instance.generate(input_ids, max_length=100, temperature=0.7, top_k=50)

    # generation using the special generation method
    mode = 'TTT_MLP'
    ttt_experiment_kwargs = {
        'ttt_lr': .2,
        'lr_sqrt_decay': False,
        'ttt_init_K': 0.005,
        'ttt_init_V': 0.01,
        'special_layers': [1, 3, 5, 7],
        'TTT_hidden_dim': 32,
        'activation': 'sigmoid',
        'init_mode': 'normal',
    }
    special_generated_ids = model_instance.special_generate(input_ids, mode=mode, max_length=100, temperature=0.7, top_k=50, **ttt_experiment_kwargs)

    # decoding and printing the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    special_generated_text = tokenizer.decode(special_generated_ids[0], skip_special_tokens=True)
    print(f"Special Generated text ({mode}): {special_generated_text}")