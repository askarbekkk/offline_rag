import fitz  # PyMuPDF
import pdfplumber
import camelot
from typing import List
from langchain_core.documents import Document
import pandas as pd

class EnhancedPDFLoader:
    """
    Enhanced PDF loader combining multiple libraries for better extraction.
    Uses PyMuPDF for text, pdfplumber for tables, and camelot as fallback.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def extract_tables_pdfplumber(self) -> List[str]:
        """Extract tables using pdfplumber - good for most PDFs"""
        tables_markdown = []
        
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if table:
                        # Convert to DataFrame for easier markdown conversion
                        df = pd.DataFrame(table[1:], columns=table[0])
                        markdown_table = df.to_markdown(index=False)
                        tables_markdown.append(f"\n## Table {table_idx + 1} (Page {page_num})\n{markdown_table}\n")
        
        return tables_markdown
    
    def extract_tables_camelot(self) -> List[str]:
        """Extract tables using camelot - better for complex tables"""
        tables_markdown = []
        
        try:
            # Use 'lattice' for tables with borders, 'stream' for borderless
            tables = camelot.read_pdf(self.file_path, pages='all', flavor='lattice')
            
            for idx, table in enumerate(tables):
                df = table.df
                markdown_table = df.to_markdown(index=False)
                tables_markdown.append(f"\n## Table {idx + 1}\n{markdown_table}\n")
                
        except Exception as e:
            print(f"Camelot extraction failed: {e}")
            
        return tables_markdown
    
    def extract_text_pymupdf(self) -> List[Dict]:
        """Extract text with layout preservation using PyMuPDF"""
        pages_content = []
        
        doc = fitz.open(self.file_path)
        for page_num, page in enumerate(doc, 1):
            # Get text with layout preservation
            text = page.get_text("text")
            
            # Get images info
            images = page.get_images()
            
            pages_content.append({
                'page_number': page_num,
                'text': text,
                'has_images': len(images) > 0,
                'image_count': len(images)
            })
        
        doc.close()
        return pages_content
    
    def load(self) -> List[Document]:
        """
        Main loading method that combines text and table extraction
        """
        documents = []
        
        # Extract text from all pages
        pages = self.extract_text_pymupdf()
        
        # Try to extract tables
        try:
            tables = self.extract_tables_pdfplumber()
        except:
            try:
                tables = self.extract_tables_camelot()
            except:
                tables = []
        
        # Create documents
        for page_data in pages:
            page_content = page_data['text']
            
            # Add table information if available
            page_tables = [t for t in tables if f"Page {page_data['page_number']}" in t]
            if page_tables:
                page_content += "\n\n" + "\n".join(page_tables)
            
            doc = Document(
                page_content=page_content,
                metadata={
                    'source': self.file_path,
                    'page': page_data['page_number'],
                    'has_images': page_data['has_images'],
                    'image_count': page_data['image_count'],
                    'has_tables': len(page_tables) > 0
                }
            )
            documents.append(doc)
        
        return documents


# Install required packages:
# pip install pdfplumber camelot-py[cv] opencv-python-headless ghostscript