import os
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import numpy as np
import pytesseract
from googletrans import Translator

# --------- CONFIG: Set these paths ---------
INPUT_PDF_PATH = "D:/SchemeNavData/Pdf/cmsclorship.pdf"
OUTPUT_TXT_PATH = "D:/SchemeNavData/output.txt"
POPPLER_PATH = r"D:/Release-24.08.0-0/poppler-24.08.0/Library/bin"
TESSERACT_PATH = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# --------- ENVIRONMENT: Only if not in PATH ---------
os.environ['PATH'] = POPPLER_PATH + os.pathsep + os.environ.get('PATH', '')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --------- IMAGE PREPROCESSING ---------
def preprocess_image(image):
    try:
        gray = image.convert('L')
        denoised = gray.filter(ImageFilter.MedianFilter(size=3))
        arr = np.array(denoised)
        threshold = arr.mean()
        binarized = (arr > threshold) * 255
        out_img = Image.fromarray(binarized.astype(np.uint8))
        return out_img
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image

# --------- PDF TO IMAGES ---------
def pdf_to_images(pdf_path):
    try:
        return convert_from_path(pdf_path)
    except Exception as e:
        print(f"PDF to image failed: {e}")
        return []

# --------- OCR (Kannada) ---------
def ocr_image(image):
    try:
        return pytesseract.image_to_string(image, lang='kan')
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

# --------- CHUNKING FOR TRANSLATION ---------
def split_text(text, max_chunk_len=2000):
    # Google Translate supports larger chunks than NMT models
    sentences = text.split('\n')
    batches, current_batch = [], ''
    for sentence in sentences:
        if len(current_batch) + len(sentence) + 1 <= max_chunk_len:
            current_batch += sentence + '\n'
        else:
            batches.append(current_batch.strip())
            current_batch = sentence + '\n'
    if current_batch:
        batches.append(current_batch.strip())
    return batches

# --------- TRANSLATION USING GOOGLETRANS ---------
def translate_to_english(chunks):
    translator = Translator()
    translations = []
    for chunk in chunks:
        try:
            # Googletrans sometimes fails if string is empty
            text_to_translate = chunk.strip()
            if text_to_translate:
                translation = translator.translate(text_to_translate, src='kn', dest='en')
                translations.append(translation.text)
            else:
                translations.append('')
        except Exception as e:
            print(f"Translation error: {e}")
            translations.append('')
    return translations

# --------- SAVE OUTPUT ---------
def save_text(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved output to {output_path}")
    except Exception as e:
        print(f"Failed to save output: {e}")

# --------- PIPELINE EXECUTION ---------
if __name__ == "__main__":
    images = pdf_to_images(INPUT_PDF_PATH)
    kannada_text = ""
    for i, img in enumerate(images):
        try:
            processed_img = preprocess_image(img)
            ocr_txt = ocr_image(processed_img)
            kannada_text += ocr_txt + "\n"
        except Exception as e:
            print(f"Page {i+1} processing error: {e}")

    batches = split_text(kannada_text, max_chunk_len=2000)  # googletrans supports larger chunks
    eng_chunks = translate_to_english(batches)
    final_english = "\n".join(eng_chunks)
    save_text(final_english, OUTPUT_TXT_PATH)
