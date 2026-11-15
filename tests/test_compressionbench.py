import pytest
import torch
from compressionbench import CompressionBench, create_compressionbench

class TestCompressionBench:
    def test_init(self):
        m = CompressionBench(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = CompressionBench(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_compressionbench()
        assert isinstance(m, CompressionBench)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
