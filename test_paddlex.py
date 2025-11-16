import os
from datetime import timedelta
from time import perf_counter

os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
os.environ["PADDLE_PDX_CACHE_HOME"] = r"C:\Users\Victor\Downloads\temp models"

from test_paddleocr import PaddleOCR

start_time = perf_counter()

img_files = r"C:\Users\Victor\OneDrive\Public\test images"

ocr_fn = PaddleOCR(
    lang="it",
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
)
results = ocr_fn.predict_iter(img_files)
for res in results:
    print(res)
    # res.save_to_img("output")
    # res.save_to_json("output")
    print("-" * 200)
    # break

print(f"Duration: {timedelta(seconds=round(perf_counter() - start_time))}")
