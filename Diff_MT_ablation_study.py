"""Component ablation under identical cross-validation splits."""
from train import parser, run

if __name__ == '__main__':
    run(parser(('full', 'no_dmfe', 'no_gce', 'no_dwf')).parse_args())
