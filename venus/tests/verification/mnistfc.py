from venus.verification.venus import Venus
from venus.common.parameters import Params

params = Params()
params.set_param('complete', True)
venus = Venus(
    nn='venus/tests/data/mnistfc/nets/mnist-net_256x2.onnx',
    spec='venus/tests/data/mnistfc/specs/0.03',
    params=params
)
results = venus.verify()
