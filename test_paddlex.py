import os
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter

os.environ["PADDLE_PDX_CACHE_HOME"] = r"C:\Users\Victor\Downloads\temp models"

from test_paddleocr import PaddleOCR, TextDetection, TextRecognition

start_time = perf_counter()

ocr = PaddleOCR()
text_det = TextDetection()
text_rec = TextRecognition()

img_files = Path(r"C:\Users\Victor\OneDrive\Public\test images")

for img_file in img_files.iterdir():
    print(img_file)
    result = ocr.predict(str(img_file))
    print(result)
    result2 = text_det.predict(str(img_file))
    print("-" * 200)
    print(result2)
    result3 = text_rec.predict(str(img_file))
    print("-" * 200)
    print(result3)
    print("=" * 200)
    break

subprocess.run(["paddlex", "--install", "paddle2onnx"])

print(f"Duration: {timedelta(seconds=round(perf_counter() - start_time))}")
