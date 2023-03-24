import time
import torch
from tqdm.auto import tqdm
from model import GPT, GPTConfig
from data.prepare_data import get_dataloader

def test_model(config_path="config/medium.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model from {config_path}.")
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    model.to(device)
    
    print("Testing model forward pass.")
    example_input = torch.randint(0, config.vocab_size, (4, config.seq_len))
    example_input = example_input.to(device)
    start = time.time()
    out1 = model(example_input)
    print("Forwarded initial batch in", time.time() - start, "seconds")
    assert out1.shape == (4, config.seq_len, config.vocab_size), "Output shape is incorrect."
    # delete to save memory
    del out1

    print("Model forward pass test passed.")

def test_compiled_model(config_path="config/medium.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model from {config_path}.")
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    model.to(device)
    compiled = torch.compile(model, mode="max-autotune")
    print("torch.compile() ran successfully.")
    print("Device of compiled model:", compiled.named_parameters().__next__()[1].device)

    print("Testing initial model forward pass.")
    example_input = torch.randint(0, config.vocab_size, (4, config.seq_len))
    example_input = example_input.to(device)
    start = time.time()
    out1 = compiled(example_input)
    print("Forwarded first batch in", time.time() - start, "seconds")
    assert out1.shape == (4, config.seq_len, config.vocab_size), "Output shape is incorrect."
    # delete to save memory
    del out1
    print("Compiled model test passed.")

# from pytorch docs
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def run_forward_passes(config, model, device, num_passes=100):
    times = []
    for i in tqdm(range(num_passes)):
        random_tensor = torch.randint(0, config.vocab_size, (4, config.seq_len))
        random_tensor = random_tensor.to(device)
        out, time = timed(lambda: model(random_tensor))
        times.append(time)
        del out
    return times

def test_compiled_speedup(config_path="config/medium.yaml"):
    # Create model and compiled model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model from {config_path}.")
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    model.to(device)
    compiled = torch.compile(model, mode="max-autotune")
    print("torch.compile() ran successfully.")
    
    # Run forward passes on normal model
    print("Warming up...")
    run_forward_passes(config, model, device, num_passes=100)
    print("Testing normal forward pass.")
    normal_times = run_forward_passes(config, model, device)
    print(torch.mean(torch.tensor(normal_times)).item())

    # Run forward passes on compiled model
    print("Warming up...")
    run_forward_passes(config, compiled, device, num_passes=100)
    print("Testing compiled forward pass.")
    compiled_times = run_forward_passes(config, compiled, device)
    print(torch.mean(torch.tensor(compiled_times)).item())
    
def test_dataloader():
    loader = get_dataloader()
    i = 0
    for X, Y in loader:
        print(X.shape, Y.shape)
        i += 1
        if i > 10:
            break
        

if __name__ == "__main__":
    #test_model()
    if torch.cuda.is_available():
        test_compiled_model()
        test_compiled_speedup()
    else:
        print("Skipping compiled model tests because CUDA is not available.")
    test_dataloader()