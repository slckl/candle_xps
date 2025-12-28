import sys

import onnx

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_onnx_file>")
    sys.exit(1)

model = onnx.load(sys.argv[1])
print(onnx.helper.printable_graph(model.graph))
