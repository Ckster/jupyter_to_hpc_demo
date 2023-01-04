# Imports
import os

# Classes / Functions / Constants
PROJECT_DIR = os.path.join(os.path.sep, 'projects', os.environ['USER'], 'my_project')
DATA_IN = os.path.join(PROJECT_DIR, 'data_in')
DATA_OUT = os.path.join(PROJECT_DIR, 'data_out', os.environ['SLURM_JOB_ID'])


# Main
def main():
    os.makedirs(DATA_OUT, exist_ok=True)


# Hook
if __name__ == '__main__':
    main()
