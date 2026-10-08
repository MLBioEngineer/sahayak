import sys
import logging
from services.rag_service import ingest_documents

# Set up simple logging for the console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    print("========================================")
    print(" Sahayak AI - Medical Document Ingestion ")
    print("========================================")
    print("Scanning 'backend/medical_docs' for PDF files...")
    
    success = ingest_documents()
    
    if success:
        print("\n✅ Ingestion complete! The Vector Database has been updated.")
        print("Your Sahayak AI chatbot will now use these documents for context.")
    else:
        print("\n⚠️ Ingestion skipped or failed.")
        print("Please ensure you have placed PDF files in 'backend/medical_docs/'")
        print("and installed the required packages (pip install -r requirements.txt).")
        sys.exit(1)
