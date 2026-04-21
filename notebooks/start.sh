#!/bin/bash

# Set CUDA library paths before launching Jupyter
NVIDIA_LIBS=$(python -c "
import site, glob, os
libs = []
for sp in site.getsitepackages():
    libs.extend(glob.glob(os.path.join(sp, 'nvidia', '*', 'lib')))
print(':'.join(libs))
")

if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi

jupyter notebook "$@"   # passes through any extra args you add