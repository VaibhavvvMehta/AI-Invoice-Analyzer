from PIL import Image
import pytesseract


try:
    img = Image.open("sample_invoice.jpg")
    print("✅ Image loaded successfully.")
    
    text = pytesseract.image_to_string(img)
    print("📄 Extracted Text:\n")
    print(text)
except Exception as e:
    print(f"⚠️ An error occurred: {e}")
