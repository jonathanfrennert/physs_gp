from .kernel import Kernel, MarkovKernel


class ApproximateMarkovKernel(MarkovKernel):
    def __init__(self, kernel: Kernel):
        super(ApproximateMarkovKernel, self).__init__()

        self.base_kernel = kernel
