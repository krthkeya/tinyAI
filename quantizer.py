# This code handles downloading, quantizing and saving the model
# takes a model from hugging face, saves gguf and quantized gguf files

import subprocess
import logging
import os

class quantized_model_creator:
    
    def __init__(self, hugging_face_url, python_interpreter, package_manager, fp_format, quantization_format):
        """
        Parameters:
        hugging_face_url (str): the url of the model on hugging face.
        python_interpreter (str): the python interpreter on the device genereally either python or python3.
        package_manager (str): the package manager on the device generally pip or pip3.
        fp_format (str): the format to store the gguf in example - fp16.
        quantization_format (str): the quantization format to be used example - Q4_K_M.
        """
        # setting up logging 
        logging.basicConfig(
            filename="quantizer.log",
            level = logging.DEBUG,
            format = "%(levelname)s - %(message)s - %(asctime)s"
        )

        #loading into class attributes
        self.hugging_face_url = hugging_face_url
        self.model_dir = self.hugging_face_url.split("/")[-1]
        self.python_interpreter = python_interpreter
        self.package_manager = package_manager
        self.fp_format = fp_format
        self.quantization_format = quantization_format

    def download_model(self):
        # downloading the model
        subprocess.run("git lfs install", shell=True)
        subprocess_attempt = subprocess.run(["git", "clone", self.hugging_face_url], capture_output=True, text=True)
        logging.info(f"model download attempt : {subprocess_attempt}")
    
    def convert_to_gguf(self):
        if not os.path.exists("llama.cpp"):
            subprocess_attempt = subprocess.run("git clone https://github.com/ggml-org/llama.cpp", shell=True, capture_output=True, text=True)
            logging.info(f"llama.cpp clone attempt : {subprocess_attempt}")
        subprocess_attempt = subprocess.run([self.package_manager,  "install" ,"-r", "requirements.txt"], cwd = "llama.cpp", capture_output=True)
        logging.info(f"requirements installation attempt : {subprocess_attempt}")
        subprocess_attempt = subprocess.run(["mkdir", f"GGUF_{self.model_dir}"], text=True, capture_output=True)
        logging.info(subprocess_attempt)
        subprocess_attempt = subprocess.run([self.python_interpreter, "llama.cpp/convert_hf_to_gguf.py", f"./{self.model_dir}/", "--outtype", f"{self.fp_format}", "--outfile", f"./GGUF_{self.model_dir}/{self.fp_format}.gguf"], capture_output=True)
        logging.info(f"HF -> GGUF attempted {subprocess_attempt}")
    
    def quantize_gguf(self):
        if not os.path.exists("llama.cpp/build"):
            subprocess_attempt = subprocess.run(["cmake", "-B", "build"], capture_output=True, text=True, cwd="llama.cpp")
            logging.info(f"build attempt {subprocess_attempt}")
            subprocess_attempt = subprocess.run(["cmake", "--build", "build", "--config", "Release"], capture_output=True, text=True, cwd="llama.cpp")
            logging.info(f"build attempt {subprocess_attempt}")
        subprocess_attempt = subprocess.run(["llama.cpp/build/bin/llama-quantize", f"./GGUF_{self.model_dir}/{self.fp_format}.gguf", f"./GGUF_{self.model_dir}/{self.quantization_format}.gguf", f"{self.quantization_format}"], capture_output=True, text=True)
        logging.info(f"quantization attempt {subprocess_attempt}")
    
    def get_quantized_model_path(self):
        if os.path.exists("GGUF_" + self.model_dir + "/" + self.quantization_format + ".gguf"):
            return "GGUF_" + self.model_dir + "/" + self.quantization_format + ".gguf"
        else:
            raise Exception("failed to find quantized model path")

# Below is a sample usage of the code   


# my_intelligent_model = quantized_model_creator(
#     hugging_face_url="https://huggingface.co/Qwen/Qwen2.5-0.5B",
#     python_interpreter = "python3",
#     package_manager = "pip3",
#     fp_format="f16",
#     quantization_format = "Q4_K_M"
# )
# my_intelligent_model.download_model()
# my_intelligent_model.convert_to_gguf()
# my_intelligent_model.quantize_gguf()
# print(my_intelligent_model.get_quantized_model_path())
