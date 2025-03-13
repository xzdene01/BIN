import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CGP circuit.")
    parser.add_argument("-f", "--file", type=str, required=True, help="The file to load the CGP circuit from.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-g", "--generate", action="store_true", help="Generate all tests (triplets of two inputs and one output).")
    group.add_argument("-s", "--source_file", type=str, help="The file from which to load tests (triplets of two inpits and one output).")

    parser.add_argument("-b", "--batch_size", type=int, default=16, help="The batch size used during generation and inference, will be converted to 2 ** batch_size (default: 16).")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], default=None, help="The device to use during inference (if not provided cuda will be tried).")
    return parser.parse_args()