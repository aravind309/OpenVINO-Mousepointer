import os
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore


class Inference:
    def __init__(self):
        self.IECore = IECore()
        self.network = None
        self.inputBlob = None
        self.outputBlob = None
        self.execNetwork = None
        self.infer_request = None

    def loadModel(self, model, device="CPU", num_requests=0):

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Read the IR as a IENetwork
        try:
            self.network =  self.IECore.read_network(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Enter correct Path and make sure models to have models in the entered directory")

        # Check Network layer support
        if "CPU" in device:
            supported_layers = self.IECore.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                sys.exit(1)

        # Load the IENetwork into the plugin
        self.execNetwork = self.IECore.load_network(self.network, device, num_requests=num_requests)

        # Get the input layer
        self.inputBlob = next(iter(self.network.inputs))
        self.outputBlob = next(iter(self.network.outputs))
        return

    def preprocess_input(self, image):
        n, c, h, w = self.get_input_shape()
        transformedImage = cv2.resize(image, (w, h))
        transformedImage = transformedImage.transpose((2, 0, 1))
        transformedImage = transformedImage.reshape((n, c, h, w))

        return transformedImage

    def get_input_shape(self):
        return self.network.inputs[self.inputBlob].shape

    def get_gaze_input_shape(self):
        self.inputBlob = [i for i in self.network.inputs.keys()]
        self.outputBlob = [i for i in self.network.outputs.keys()]

        return self.network.inputs[self.inputBlob[1]].shape

    def prediction_helper(self, image, preprocess_output, prob_threshold=0.6):
        processedImage = self.preprocess_input(image.copy())
        result = self.execNetwork.infer(inputs={self.inputBlob: processedImage})
        coords = preprocess_output(result, prob_threshold)

        return coords

