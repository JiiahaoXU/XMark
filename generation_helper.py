def get_generation_kwargs(args):
    gen_kwargs = dict(max_new_tokens=args.tokens_for_detection_per_prompt, min_new_tokens=args.tokens_for_detection_per_prompt)

    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))
    
    return gen_kwargs


def get_token_input_ids(args, model, tokenizer, prompt, device):
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.tokens_for_detection_per_prompt
    else:
        args.prompt_max_length = 2048-args.tokens_for_detection_per_prompt
    
    token_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                            max_length=args.prompt_max_length).to(device)
        
    return token_inputs