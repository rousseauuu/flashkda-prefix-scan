"""Expose launch stages without changing the vendored CUDA kernel bodies."""
from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(path, old, new):
    source = path.read_text()
    assert old in source, (path, old)
    path.write_text(source.replace(old, new))


for path in (root / 'csrc/fwd.h', root / 'csrc/smxx/fwd_launch.cu'):
    replace(path, 'cudaStream_t stream\n', 'cudaStream_t stream,\n    int stage\n')
launch = root / 'csrc/smxx/fwd_launch.cu'
replace(launch, '#if BLOCK_LEVEL_K1 >= 0\n    {', '#if BLOCK_LEVEL_K1 >= 0\n    if (stage != 2) {')
replace(launch, '#if BLOCK_LEVEL_K2 >= 0\n    {', '#if BLOCK_LEVEL_K2 >= 0\n    if (stage != 1) {')
replace(launch, 'float, cudaStream_t);', 'float, cudaStream_t, int);')
binding = root / 'csrc/flash_kda.cpp'
replace(binding, 'std::optional<torch::Tensor> cu_seqlens = std::nullopt\n',
        'std::optional<torch::Tensor> cu_seqlens = std::nullopt,\n    int stage = 0\n')
replace(binding, 'gate_scale, stream)', 'gate_scale, stream, stage)')
replace(binding, 'py::arg("cu_seqlens") = py::none());',
        'py::arg("cu_seqlens") = py::none(), py::arg("stage") = 0);')
print('Added stage=0 (original), 1 (K1), 2 (K2); CUDA kernel bodies unchanged.')
