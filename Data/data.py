import os
from pdf2image import convert_from_path
import google.generativeai as genai
from dotenv import load_dotenv

# ------------ Configuration ------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_FOLDER = "D:/SchemeNavData/Pdf"
POPPLER_PATH = r"D:/Release-24.08.0-0/poppler-24.08.0/Library/bin"
OUTPUT_FOLDER = "D:/SchemeNavData/Translated"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------ Gemini Setup ------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

# ------------ Prompt ------------
prompt = """
You are an OCR and translation expert. The image is a scanned Kannada government document page.
Please:
1. Perform OCR to extract Kannada text.
2. Fix any OCR errors.
3. Translate the corrected Kannada text to fluent English.
4. Output both the corrected Kannada and English translations.
"""

# ------------ Process PDFs ------------
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

        print(f"\nProcessing: {filename}")

        try:
            # Convert all pages
            images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

            full_output = ""

            for i, image in enumerate(images, start=1):
                image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_page{i}.png")
                image.save(image_path, "PNG")

                print(f"  - Gemini OCR+Translate Page {i}/{len(images)}")
                try:
                    response = model.generate_content([prompt, genai.upload_file(image_path)])
                    page_text = response.text.strip() if response.text else "[No text]"
                    full_output += f"\n--- Page {i} ---\n{page_text}\n"
                except Exception as e:
                    print(f"   Error on page {i}: {e}")
                    full_output += f"\n--- Page {i} ---\n[Error: {e}]\n"

            # Save the full output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_output)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
