"""PDF processing module for medical documents."""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
import spacy
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a medical document."""
    title: str
    content: str
    page_number: int
    start_char: int
    end_char: int
    is_bold: bool = False
    font_size: Optional[float] = None

@dataclass
class ProcessedDocument:
    """Represents a processed medical document."""
    filename: str
    sections: List[DocumentSection]
    full_text: str
    metadata: Dict

class PDFProcessor:
    """Handles PDF extraction, section segmentation, and text cleaning."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the PDF processor."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Spacy model {spacy_model} not found. Using basic tokenizer.")
            self.nlp = None
            
        # Regex patterns for cleaning
        self.page_number_pattern = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
        self.header_footer_pattern = re.compile(r'^(page\s+\d+|\d+\s+of\s+\d+|doi:|http://|https://).*$', re.MULTILINE | re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\']')
        
    def extract_text_with_formatting(self, pdf_path: Path) -> List[Dict]:
        """Extract text with formatting information using PyMuPDF."""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")
            page_data = {
                'page_number': page_num + 1,
                'blocks': []
            }
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_fonts = []
                        line_sizes = []
                        line_bold = []
                        
                        for span in line["spans"]:
                            text = span["text"]
                            font = span["font"]
                            size = span["size"]
                            
                            line_text += text
                            line_fonts.append(font)
                            line_sizes.append(size)
                            line_bold.append("bold" in font.lower())
                        
                        if line_text.strip():
                            page_data['blocks'].append({
                                'text': line_text,
                                'fonts': line_fonts,
                                'sizes': line_sizes,
                                'is_bold': any(line_bold),
                                'avg_size': sum(line_sizes) / len(line_sizes) if line_sizes else 0,
                                'bbox': line["bbox"]
                            })
            
            pages_data.append(page_data)
        
        doc.close()
        return pages_data
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber as fallback."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def detect_sections(self, pages_data: List[Dict]) -> List[DocumentSection]:
        """Detect document sections based on formatting changes."""
        sections = []
        current_section = None
        current_content = []
        
        for page_data in pages_data:
            for block in page_data['blocks']:
                text = block['text'].strip()
                is_bold = block['is_bold']
                avg_size = block['avg_size']
                
                # Skip empty blocks
                if not text:
                    continue
                
                # Detect potential headings (bold text, larger font, short lines)
                is_potential_heading = (
                    is_bold and 
                    len(text) < 100 and 
                    avg_size > 10 and
                    not text.endswith('.') and
                    not re.match(r'^\d+\.?\s*$', text)  # Not just numbers
                )
                
                if is_potential_heading and current_section:
                    # Save current section
                    if current_content:
                        current_section.content = '\n'.join(current_content)
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = DocumentSection(
                        title=text,
                        content="",
                        page_number=page_data['page_number'],
                        start_char=0,  # Will be updated later
                        end_char=0,    # Will be updated later
                        is_bold=is_bold,
                        font_size=avg_size
                    )
                    current_content = []
                else:
                    # Add to current content
                    if current_section is None:
                        current_section = DocumentSection(
                            title="Introduction",
                            content="",
                            page_number=page_data['page_number'],
                            start_char=0,
                            end_char=0,
                            is_bold=False
                        )
                    current_content.append(text)
        
        # Add final section
        if current_section and current_content:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
        
        return sections
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing headers, footers, and normalizing whitespace."""
        # Remove page numbers
        text = self.page_number_pattern.sub('', text)
        
        # Remove headers and footers
        text = self.header_footer_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove excessive special characters
        text = self.special_chars_pattern.sub('', text)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def create_chunks_with_boundaries(self, text: str) -> List[Dict]:
        """Create chunks while preserving sentence boundaries."""
        if self.nlp:
            # Use spacy for sentence boundary detection
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk,
                    'start_char': current_start,
                    'end_char': current_start + len(current_chunk),
                    'length': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_start += len(current_chunk) - len(sentence) - (len(overlap_text) + 1 if overlap_text else 0)
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append({
                'text': current_chunk,
                'start_char': current_start,
                'end_char': current_start + len(current_chunk),
                'length': len(current_chunk)
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> ProcessedDocument:
        """Main method to process a PDF file."""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Try PyMuPDF first for formatting information
            pages_data = self.extract_text_with_formatting(pdf_path)
            
            # Extract full text
            full_text = ""
            for page_data in pages_data:
                for block in page_data['blocks']:
                    full_text += block['text'] + "\n"
            
            # Detect sections
            sections = self.detect_sections(pages_data)
            
            # Clean text
            full_text = self.clean_text(full_text)
            
            # Create chunks
            chunks = self.create_chunks_with_boundaries(full_text)
            
            # Prepare metadata
            metadata = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'total_pages': len(pages_data),
                'total_sections': len(sections),
                'total_chunks': len(chunks),
                'file_size': pdf_path.stat().st_size,
                'processing_method': 'PyMuPDF + pdfplumber'
            }
            
            return ProcessedDocument(
                filename=pdf_path.name,
                sections=sections,
                full_text=full_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}. Falling back to pdfplumber.")
            
            # Fallback to pdfplumber
            full_text = self.extract_text_pdfplumber(pdf_path)
            full_text = self.clean_text(full_text)
            
            # Create a single section for fallback
            sections = [DocumentSection(
                title="Document",
                content=full_text,
                page_number=1,
                start_char=0,
                end_char=len(full_text),
                is_bold=False
            )]
            
            chunks = self.create_chunks_with_boundaries(full_text)
            
            metadata = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'total_pages': 1,
                'total_sections': 1,
                'total_chunks': len(chunks),
                'file_size': pdf_path.stat().st_size,
                'processing_method': 'pdfplumber_fallback'
            }
            
            return ProcessedDocument(
                filename=pdf_path.name,
                sections=sections,
                full_text=full_text,
                metadata=metadata
            )
