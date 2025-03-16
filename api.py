from flask import Flask, request, jsonify
from quantizer import quantized_model_creator
from inference import language_model_inference

app = Flask(__name__)

@app.route("/create_quantized_model", methods = ["POST"])
def create_quantized_model():
    try:
        data = request.json
        model_creator = quantized_model_creator(
            hugging_face_url=data.get("hugging_face_url"),
            python_interpreter = data.get("python_interpreter"),
            package_manager = data.get("package_manager"),
            fp_format=data.get("fp_format"),
            quantization_format = data.get("quantization_format")
        )
        model_creator.download_model()
        model_creator.convert_to_gguf()
        model_creator.quantize_gguf()
        return jsonify(
            {
                "success" : True,
                "message" : model_creator.get_quantized_model_path()
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success" : False,
                "message" : str(e)
            }
        )
@app.route("/heyllm", methods = ["GET"])
def heyllm():
    try:
        model_path, prompt, tokens = request.args.get("model_path"), request.args.get("prompt"), request.args.get("tokens")
        llm_instance = language_model_inference(model_path=model_path)
        llm_response = llm_instance.hey_llm(prompt=prompt, tokens=tokens)
        return jsonify(
            {
                "success" : True,
                "message" : llm_response
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success" : False,
                "message" : str(e)
            }
        )

if __name__ == "__main__":
    app.run(debug=True)