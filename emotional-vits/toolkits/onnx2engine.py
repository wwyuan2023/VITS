# coding: utf-8
#
# python api: onnx转engine
#

import sys
import os
import argparse
import tensorrt as trt

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def main():
    
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("--input", "-i", type=str, required=True, help="input onnx model path.")
    parser.add_argument("--output", "-o", type=str, required=True, help="output trt engine path.")
    
    args = parser.parse_args()
    onnx_path = args.input
    engine_path = args.output
    
    print('get start')
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
        
        print(f"raw shape of {network.get_input(0).name} is: ", network.get_input(0).shape)
        print(f"raw shape of {network.get_input(1).name} is: ", network.get_input(1).shape)
        print(f"raw shape of {network.get_input(2).name} is: ", network.get_input(2).shape)
        print(f"raw shape of {network.get_input(3).name} is: ", network.get_input(3).shape)
        print(f"raw shape of {network.get_input(4).name} is: ", network.get_input(4).shape)
        print(f"raw shape of {network.get_input(5).name} is: ", network.get_input(5).shape)
        print(f"raw shape of {network.get_output(0).name} is: ", network.get_output(0).shape)

        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 30) * 1  # 1 GB
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, (1, 3, 192), (1, 128, 192), (1, 256, 192))
        profile.set_shape(network.get_output(0).name, (1, 1, 2000), (1, 1, 50000), (1, 1, 160000))
        config.add_optimization_profile(profile)

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_path))
        engine_serialize = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        with open(engine_path, "wb") as f:
            f.write(engine_serialize)


if __name__ == "__main__":
    
    main()
    