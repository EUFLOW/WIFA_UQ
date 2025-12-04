import sys
import os
from pathlib import Path

# --- Add the project root to the Python path ---
# This ensures that the script can find the 'wifa_uq' package
# when you run `python run.py ...` from the root directory.
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

#try:
from wifa_uq.workflow import run_workflow
#except ImportError:
#    print("Error: Could not import 'run_workflow' from 'wifa_uq.workflow'.")
#    print("Please ensure:")
#    print(f"  1. You are running this script from the project root: {project_root}")
#    print("  2. The 'wifa_uq' package exists and 'workflow.py' is inside it.")
#    sys.exit(1)

def main():
    """
    Main entry point for running a WIFA-UQ workflow.
    """
    # Check if a config file path was provided as an argument
    if len(sys.argv) < 2:
        print("Error: No configuration file specified.")
        print("Usage: python run.py path/to/your_config.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # --- Validate Config Path ---
    if not config_path.exists():
        print(f"Error: Configuration file not found at: {config_path.resolve()}")
        sys.exit(1)
    
    # --- Run the Workflow ---
    print(f"--- Starting WIFA-UQ Workflow ---")
    print(f"Using config file: {config_path.resolve()}")
    
    run_workflow(config_path)
    print(f"--- Workflow Finished Successfully ---")
        

if __name__ == "__main__":
    main()
