from flask import Flask
from flask_restful import Resource, Api
import numpy as np
import tflite_runtime.interpreter as tflite
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "gpt-vocab.json",
    "gpt-merges.txt"
)

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        print('hello world')
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/')

# API to infer text with gpt2


class InferText(Resource):
    def get(self, text):
        print('text: ', text)
        output_data = infer_text(text)
        return {'text': output_data}


api.add_resource(InferText, '/infer/<string:text>')
if __name__ == '__main__':
    app.run(debug=True)

# function to infer text with gpt2-small.tflite


def infer_text(text):
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="gpt2-small.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Tokenize input text
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids

    # Convert input_ids to int32
    input_ids = np.array(input_ids, dtype=np.int32)

    # Run the model
    interpreter.set_tensor(input_details[0]["index"], input_ids)
    interpreter.invoke()
    output_ids = interpreter.get_tensor(output_details[0]["index"])

    # Decode the output
    output_text = tokenizer.decode(output_ids[0])

    return output_text
