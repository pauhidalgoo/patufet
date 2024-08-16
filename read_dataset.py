from datasets import load_dataset
ds = load_dataset("uonlp/CulturaX",
                  "ca",
                  use_auth_token=True)