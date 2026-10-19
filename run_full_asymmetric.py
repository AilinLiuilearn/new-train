"""Run the same Full-only protocol with PET-DSSA/CT-EGA fusion."""
from train_full import main

if __name__=='__main__':
    main(default_fusion='asymmetric')
