from watermark_methods.mpac.mpac import MPACProcessor, MPACDetector
from watermark_methods.rsbh.rsbh import RSBHProcessor, RSBHDetector
from watermark_methods.cycleshift import CycleShiftProcessor, CycleShiftDetector
from watermark_methods.depthd import DepthWProcessor, DepthWDetector
from watermark_methods.stealthink.stealthink import SteathInkProcessor, SteathInkDetector
from watermark_methods.loso.loso import LOSOProcessor, LOSODetector
from watermark_methods.xmark import XMARKProcessor, XMARKDetector


def get_processor(wm_method, args, tokenizer, current_binary_code, device):
    if wm_method == 'mpac':
        
        wm_kwargs = {
            'use_position_prf': False,
            'use_fixed_position': False,
            'code_length': args.bits,
            'use_feedback': False,
            'feedback_args': {'eta': 2,
                              'tau': 2,
                              'feedback_bias': 2
                              }
                 }
        watermark_processor = MPACProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            base=2,
            seeding_scheme="simple_1",
            store_spike_ents=True,
            select_green_tokens=True,
            message_length=args.bits,
            device=device,
            **wm_kwargs
        )
        watermark_processor.set_message(current_binary_code)
        
    elif wm_method == 'loso':
        
        wm_kwargs = {
            'use_position_prf': False,
            'use_fixed_position': False,
            'code_length': args.bits,
            'use_feedback': False,
            'feedback_args': {'eta': 2,
                              'tau': 2,
                              'feedback_bias': 2
                              }
                 }
        watermark_processor = LOSOProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            base=2,
            seeding_scheme="simple_1",
            store_spike_ents=True,
            select_green_tokens=True,
            message_length=args.bits,
            device=device,
            **wm_kwargs
        )
        watermark_processor.set_message(current_binary_code)
        
    elif wm_method == 'xmark':
        watermark_processor = XMARKProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                           delta=args.delta,
                                                           bits=args.bits,
                                                           current_binary_code=current_binary_code,
                                                           args=args
                                                           )
        
    elif wm_method == 'rsbh':
        payload = int(current_binary_code, 2)
        
        segment_bit = 4
        segments_num=args.bits // segment_bit
        gf_segments_num=segments_num + 2
        watermark_processor = RSBHProcessor(vocab=list(tokenizer.get_vocab().values()), delta=args.delta, payload=payload, gamma=0.5,
                                                         segments_num=segments_num, gf_segments_num=gf_segments_num, segment_bit=segment_bit)

    elif wm_method == 'cycleshift':
        payload = int(current_binary_code, 2)
        
        watermark_processor = CycleShiftProcessor(vocab=list(tokenizer.get_vocab().values()), delta=args.delta, payload=payload, bits=args.bits, gamma=0.25)

    elif wm_method == 'depthw':
        watermark_processor = DepthWProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                                   gamma=args.gamma,
                                                                   decrease_delta=False,
                                                                   delta=args.delta,
                                                                   wm_mode='privious1',
                                                                   seeding_scheme="simple_1",
                                                                   select_green_tokens=True,
                                                                   userid=current_binary_code,
                                                                   args=args
                                                                   )
        
    elif wm_method == 'stealthink':
        chunk_capacity = 2
        embedded_message = [int(current_binary_code[i:i+2], 2) for i in range(0, len(current_binary_code), 2)]
        print('Embedded message:', embedded_message)
        watermark_processor = SteathInkProcessor(vocab=list(tokenizer.get_vocab().values()), chunk_capacity=chunk_capacity, embedded_message=embedded_message, args=args)
    
    else:
        raise NotImplementedError(f"Unsupported wm_method: {wm_method}")
    
    return watermark_processor

    
def get_detector(wm_method, args, tokenizer, device):
    
    if wm_method == 'mpac':
        wm_kwargs = {
            'use_position_prf': False,
            'use_fixed_position': False,
            'code_length': args.bits
        }
        watermark_detector = MPACDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme='simple_1',
            device=device,
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=args.normalizers,
            ignore_repeated_ngrams=True,
            message_length=args.bits,
            base=2,
            **wm_kwargs
        )
        
    elif wm_method == 'loso':
        wm_kwargs = {
            'use_position_prf': False,
            'use_fixed_position': False,
            'code_length': args.bits
        }
        watermark_detector = LOSODetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme='simple_1',
            device=device,
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=args.normalizers,
            ignore_repeated_ngrams=True,
            message_length=args.bits,
            base=2,
            **wm_kwargs
        )

    elif wm_method == 'xmark':
        watermark_detector = XMARKDetector(
            vocab_size=len(tokenizer),
            device=device,
            tokenizer=tokenizer,
            bits=args.bits,
            hash_key=args.hash_key,
        )
        
    elif wm_method == 'rsbh':
        
        segment_bit = 4
        segments_num=args.bits // segment_bit
        gf_segments_num=segments_num + 2

        watermark_detector = RSBHDetector(gamma=0.5, segments_num=segments_num, gf_segments_num=gf_segments_num, 
                                       segment_bit=segment_bit, device=device, 
                                       tokenizer=tokenizer, ngram=1, seeding='hash')
        
        
    elif wm_method == 'cycleshift':
        
        watermark_detector = CycleShiftDetector(gamma=0.25, bits=args.bits, device=device, 
                                       tokenizer=tokenizer, ngram=1, seeding='hash')
        
        
    elif wm_method == 'depthw':
        watermark_detector = DepthWDetector(
                                    vocab=list(tokenizer.get_vocab().values()),
                                    gamma=args.gamma,
                                    seeding_scheme="simple_1",
                                    device=device,
                                    wm_mode="combination",
                                    tokenizer=tokenizer,
                                    z_threshold=4.0,
                                    normalizers=args.normalizers,
                                    ignore_repeated_bigrams=False,
                                    select_green_tokens=True,
                                    args=args
                                )
        
    elif wm_method == 'stealthink':
        chunk_capacity = 2
        watermark_detector = SteathInkDetector(
                                    vocab=list(tokenizer.get_vocab().values()),
                                    device=device,
                                    tokenizer=tokenizer,
                                    chunk_capacity=chunk_capacity,
                                    args=args
                                )

    else:
        raise NotImplementedError(f"Unsupported wm_method: {wm_method}")
    
    return watermark_detector
    
    
def get_decoded_message(wm_method, watermark_detector, watermarked_texts):
    if wm_method == 'mpac':
        decoded_message = watermark_detector.detect(text="\n".join(watermarked_texts))
    elif wm_method == 'loso':
        decoded_message = watermark_detector.detect(text="\n".join(watermarked_texts))
    elif wm_method == 'xmark':
        decoded_message = watermark_detector.detect(watermarked_texts)
    elif wm_method == 'rsbh':
        decoded_message = watermark_detector.detect(watermarked_texts)
    elif wm_method == 'cycleshift':
        decoded_message = watermark_detector.detect(watermarked_texts)
    elif wm_method == 'depthw':
        decoded_message = watermark_detector.detect(watermarked_texts)
    elif wm_method == 'stealthink':
        decoded_message = watermark_detector.detect(watermarked_texts)
    else:
        raise NotImplementedError(f"Unsupported wm_method: {wm_method}")
    
    return decoded_message