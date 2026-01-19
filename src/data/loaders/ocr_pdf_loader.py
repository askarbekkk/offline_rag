import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from typing import List
from langchain_core.documents import Document

class OCRPDFLoader:
    """
    PDF loader with OCR support for scanned documents.
    Automatically detects if PDF needs OCR and applies it.
    """
    
    def __init__(self, file_path: str, use_ocr: bool = True):
        self.file_path = file_path
        self.use_ocr = use_ocr
        # Set tesseract path if needed (Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def is_scanned_page(self, page) -> bool:
        """
        Detect if a page is scanned (image-based) or has extractable text
        """
        text = page.get_text().strip()
        images = page.get_images()
        
        # If very little text but has images, likely scanned
        return len(text) < 50 and len(images) > 0
    
    def ocr_page(self, page) -> str:
        """
        Perform OCR on a page using Tesseract
        """
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(img_data))
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang='eng')
        
        return text
    
    def load(self) -> List[Document]:
        """
        Load PDF with automatic OCR when needed
        """
        documents = []
        doc = fitz.open(self.file_path)
        
        for page_num, page in enumerate(doc, 1):
            # Try extracting text first
            text = page.get_text()
            
            # If page appears to be scanned and OCR is enabled
            if self.use_ocr and self.is_scanned_page(page):
                print(f"Applying OCR to page {page_num}...")
                text = self.ocr_page(page)
                metadata = {
                    'source': self.file_path,
                    'page': page_num,
                    'ocr_applied': True
                }
            else:
                metadata = {
                    'source': self.file_path,
                    'page': page_num,
                    'ocr_applied': False
                }
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        doc.close()
        return documents


# Install required packages:
# pip install pytesseract pillow
# Also install Tesseract OCR:
# - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# - Linux: sudo apt-get install tesseract-ocr
# - Mac: brew install tesseract