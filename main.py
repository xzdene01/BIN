import torch
import tqdm
import time

from circuit.cgp_circuit import CGPCircuit
from utils import generator, argparser

def test_cpu(cgp, triplets, device) -> bool:
    succ = 0
    all = len(triplets)
    for in1, in2, ref in tqdm.tqdm(triplets, total=all, unit="trial"):
        input_tensor = torch.cat((in1, in2), dim=0)
        output_tensor = cgp.forward(input_tensor, device=device)
        
        if torch.equal(ref, output_tensor):
            succ += 1
    return succ, all

def test_batch(cgp, in1_tensor, in2_tensor, ref_tensor, batch_size=8192) -> bool:
    succ = 0
    all = ref_tensor.shape[0]
    for i in range(0, all, batch_size):
        batch_in1 = in1_tensor[i:i + batch_size]
        batch_in2 = in2_tensor[i:i + batch_size]
        batch_ref = ref_tensor[i:i + batch_size]

        batch_input = torch.cat((batch_in1, batch_in2), dim=1)
        batch_output = cgp.forward_batch(batch_input)

        succ += (batch_output == batch_ref).all(dim=1).sum().item()
    
    return succ

def main():
    args = argparser.parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cgp = CGPCircuit(file=args.file)

    in_bits = int(cgp.prefix["c_in"] / cgp.prefix["c_ni"])
    print(f"Bits per input: {in_bits}")

    if device == "cuda":
        batch_size = 2 ** args.batch_size
        total_trials = int(2 ** (2 * in_bits))
        total_chunks = total_trials // batch_size
        print(f"Batch size: {batch_size}")
        print(f"Total number of trials: {total_trials}")
        print(f"Total chunks: {total_chunks}")

        succ = 0
        for in1_bin, in2_bin, out_bin in tqdm.tqdm(generator.generate_all_vec(in_bits, batch_size, device=device), total=total_chunks, unit="chunk"):
            batch_input = torch.cat((in1_bin, in2_bin), dim=1)
            batch_output = cgp.forward_batch(batch_input)
            succ += (batch_output == out_bin).all(dim=1).sum().item()
        print(f"Success rate: {succ}/{total_trials} ({succ / total_trials * 100:.2f}%)")
    
    # This is like a standard CPU method, no GPU or multicore optimizations (good for up to 8 bits single inference, 4bit training)
    elif device == "cpu":
        if args.generate:
            triplets = generator.generate_all(in_bits, device=device)
            generator.save_triplets(triplets, "saved/triplets.pt")
            print("Triplets saved to file.")
        else:
            triplets = generator.load_triplets(args.source_file, device=device)
        
        succ, all = test_cpu(cgp, triplets, device=device)
        print(f"Success rate: {succ}/{all} ({succ / all * 100:.2f}%)")
    else:
        raise ValueError(f"Unknown device: {device}")


if __name__ == "__main__":
    main()