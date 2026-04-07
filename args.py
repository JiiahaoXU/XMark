import argparse


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wm_method",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2,
        help="Watermarking bias",
    )
    parser.add_argument(
        "--tokens_for_detection_per_prompt",
        type=int,
        default=75,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Main model",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=0,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )

    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        help="the number of bits to use for the watermarking",
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=50,
        help="the number of users to evaluate",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=2,
        help="the number of prompts to use for each user",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=-1,
        help="the size of the message block",
    )
    parser.add_argument(
        "--hash_key",
        type=int,
        default=15485863,
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    return args
