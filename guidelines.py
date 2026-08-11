import os
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Output directory matching your ingestion.py configuration
OUTPUT_DIR = "medical_guidelines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NCBI StatPearls Clinical Guideline URLs for your 9 dataset classes
GUIDELINE_URLS = {
    "actinic_keratosis": "https://www.ncbi.nlm.nih.gov/books/NBK430774/",
    "atopic_dermatitis": "https://www.ncbi.nlm.nih.gov/books/NBK448071/",
    "benign_keratosis": "https://www.ncbi.nlm.nih.gov/books/NBK544286/",
    "dermatofibroma": "https://www.ncbi.nlm.nih.gov/books/NBK470538/",
    "melanocytic_nevus": "https://www.ncbi.nlm.nih.gov/books/NBK538232/",
    "melanoma": "https://www.ncbi.nlm.nih.gov/books/NBK470409/",
    "squamous_cell_carcinoma": "https://www.ncbi.nlm.nih.gov/books/NBK441939/",
    "tinea_ringworm_candidiasis": "https://www.ncbi.nlm.nih.gov/books/NBK448149/",
    "vascular_lesion": "https://www.ncbi.nlm.nih.gov/books/NBK532882/",
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_and_convert_to_pdf(url, output_path):
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"Failed to fetch {url} (Status Code: {response.status_code})")
        return False

    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract main article body content
    content = soup.find("article") or soup.find("div", class_="book-ch") or soup
    
    # Configure ReportLab Document PDF
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter, 
        rightMargin=36, 
        leftMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontSize=10, leading=14, spaceAfter=6)

    story = []
    
    # Extract Title
    title_tag = soup.find("h1")
    title_text = title_tag.get_text(strip=True) if title_tag else "Clinical Guideline"
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 12))

    # Parse Headings and Paragraphs into PDF Elements
    for element in content.find_all(['h2', 'h3', 'p']):
        text = element.get_text(strip=True)
        if not text:
            continue
            
        # Clean XML entities for ReportLab rendering
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        if element.name in ['h2', 'h3']:
            story.append(Paragraph(text, heading_style))
        else:
            story.append(Paragraph(text, body_style))

    doc.build(story)
    return True

if __name__ == "__main__":
    print("Fetching clinical guidelines and compiling PDFs...\n")

    for class_name, url in GUIDELINE_URLS.items():
        pdf_path = os.path.join(OUTPUT_DIR, f"{class_name}_guideline.pdf")
        print(f"Downloading & Compiling: {class_name}...")
        success = fetch_and_convert_to_pdf(url, pdf_path)
        if success:
            print(f"  --> Saved to {pdf_path}")

    print(f"\nCompleted! 9 PDF guidelines are saved in '{OUTPUT_DIR}/'.")