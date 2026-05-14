import sys
import pandas as pd
import numpy as np

def main():
    print("=== ENVIRONMENT CHECK ===")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
    
    print("\n=== SIMPLE TEST ===")
    df = pd.DataFrame({
        "region": ["hippocampus", "entorhinal", "precuneus"],
        "value": np.random.rand(3)
    })
    
    print(df)

if __name__ == "__main__":
    main()