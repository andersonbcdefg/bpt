import time
import torch
from model import GPT, GPTConfig

def test_model(config_path="config/medium.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model from {config_path}.")
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    model.to(device)
    
    print("Testing model forward pass.")
    example_input = torch.randint(0, config.vocab_size, (8, config.seq_len))
    example_input.to(device)
    start = time.time()
    out = model(example_input)
    print("Forwarded in", time.time() - start, "seconds")

    assert out.shape == (8, config.seq_len, config.vocab_size)
    print("Model forward pass test passed.")

def test_compiled_model(config_path="config/medium.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model from {config_path}.")
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    model.to(device)
    compiled = torch.compile(model, mode="default")
    print("torch.compile() ran successfully.")
    print("Device of compiled model:", compiled.device)

    print("Testing initial model forward pass.")
    example_input = torch.randint(0, config.vocab_size, (8, config.seq_len))
    example_input.to(device)
    start = time.time()
    out1 = compiled(example_input)
    print("Forwarded first batch in", time.time() - start, "seconds")
    
    print("Testing second model forward pass.")
    start = time.time()
    out2 = compiled(example_input)
    print("Forwarded second batch in", time.time() - start, "seconds")

    assert out1.shape == (8, config.seq_len, config.vocab_size)
    assert out2.shape == (8, config.seq_len, config.vocab_size)
    print("Compiled model test passed.")
    

if __name__ == "__main__":
    test_model()
    if torch.cuda.is_available():
        test_compiled_model()
    else:
        print("Skipping compiled model test because CUDA is not available.")