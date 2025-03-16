import subprocess
import logging
import os

class language_model_inference:

    def __init__(self, model_path):
        """
        Parameters:
        model_path (str): the path to the gguf model
        """
        self.model_path = model_path
        logging.basicConfig(
            filename="inference.log",
            level=logging.DEBUG,
            format = "%(levelname)s - %(message)s - %(asctime)s"
        )

    def hey_llm(self, prompt, tokens):
        if not os.path.exists("llama.cpp"):
            subprocess_attempt = subprocess.run("git clone https://github.com/ggml-org/llama.cpp", shell=True, capture_output=True, text=True)
            logging.info(f"llama.cpp clone attempt : {subprocess_attempt}")
        if not os.path.exists("llama.cpp/build"):
            subprocess_attempt = subprocess.run(["cmake", "-B", "build"], capture_output=True, text=True, cwd="llama.cpp")
            logging.info(f"build attempt {subprocess_attempt}")
            subprocess_attempt = subprocess.run(["cmake", "--build", "build", "--config", "Release"], capture_output=True, text=True, cwd="llama.cpp")
            logging.info(f"build attempt {subprocess_attempt}")
        subprocess_attempt = subprocess.run(["llama.cpp/build/bin/llama-simple", "-m" ,self.model_path, "-n", "400", prompt], capture_output=True, text=True)
        return subprocess_attempt.stdout
    
# below is a sample usage of the code 

# llm_instance = language_model_inference(model_path="GGUF_Qwen2.5-0.5B/f16.gguf")
# print(llm_instance.hey_llm(prompt="write code binary search !", tokens="400"))
