# IKT450-remote
Code written for the IKT450 project at UiA

This is set up to be run on a remote server. Run the `run_once.py` script to download the dataset and set up the environment.

### Comments:
- I load everything into memory, as the GH200 GPU has 96GB of memory.

### Connection using Intellij:

1. Go to `Settings`
2. Go to `Project Interpreter`
3. Click `Add New Interpreter`
4. Click `On SSH`
5. Fill in the details:
    - Host: `255.255.255.255`
    - Port: `22`
    - User name: `username`
6. Click `Next`