import os
import json
import random


def load_or_generate_base_code(json_path, num_users=50, safety_block_size=4):
    """
    If the JSON file doesn't exist, generate codes for bits in [8,12,16,20,24,28,32]
    using safety_block_size=4 to enforce validity. Otherwise load and return.
    """
    bits_list = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"[Loaded] Existing code JSON found at {json_path}")
        return data["codes"]  # Return the nested dict directly

    # Otherwise: generate codes for all bits
    all_codes = {}

    for bits in bits_list:
        assert bits % safety_block_size == 0, f"Bits={bits} must be divisible by block size={safety_block_size}"
        print(f"\n[Generating] Codes for bits={bits} with block size={safety_block_size}")
        bit_codes = {}

        for user_id in range(1, 1 + num_users):
            rng = random.Random(user_id)
            num_blocks = bits // safety_block_size
            blocks = []

            for block_idx in range(num_blocks):
                while True:
                    block = ''.join(rng.choice('01') for _ in range(safety_block_size))
                    if block in {'0' * safety_block_size, '1' * safety_block_size}:
                        continue
                    blocks.append(block)
                    break

            binary_code = ''.join(blocks)
            bit_codes[str(user_id)] = binary_code
            print(f"  [User {user_id}] {binary_code}")

        all_codes[str(bits)] = bit_codes

    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump({
            "bits_list": bits_list,
            "codes": all_codes
        }, f, indent=2)

    print(f"\n✅ All codes saved to {json_path}")
    return all_codes  # Return the structure: { bits_str: {user_id: code} }