"""Main CLI interface for the Medical RAG System."""

import argparse
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.medical_rag_system import MedicalRAGSystem
except ImportError:
    # Fallback for IDE resolution
    from src.medical_rag_system import MedicalRAGSystem

def main():
    parser = argparse.ArgumentParser(description="Medical Document RAG System")
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF files to process")
    parser.add_argument("--pdf-file", type=str, help="Single PDF file to process")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--clear", action="store_true", help="Clear all indexed data")
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing Medical RAG System...")
    try:
        system = MedicalRAGSystem()
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        return 1
    
    # Process PDFs if specified
    if args.pdf_dir or args.pdf_file:
        pdf_paths = []
        
        if args.pdf_dir:
            pdf_dir = Path(args.pdf_dir)
            if not pdf_dir.exists():
                print(f"✗ Directory not found: {pdf_dir}")
                return 1
            pdf_paths.extend(list(pdf_dir.glob("*.pdf")))
        
        if args.pdf_file:
            pdf_file = Path(args.pdf_file)
            if not pdf_file.exists():
                print(f"✗ File not found: {pdf_file}")
                return 1
            pdf_paths.append(pdf_file)
        
        if not pdf_paths:
            print("✗ No PDF files found to process")
            return 1
        
        print(f"Processing {len(pdf_paths)} PDF files...")
        results = system.add_documents(pdf_paths)
        
        print(f"✓ Successfully processed: {results['total_processed']} documents")
        print(f"✓ Total chunks created: {results['total_chunks']}")
        
        if results['failed']:
            print(f"✗ Failed to process {len(results['failed'])} documents:")
            for failure in results['failed']:
                print(f"  - {failure['path']}: {failure['error']}")
    
    # Show statistics
    if args.stats:
        stats = system.get_system_stats()
        print("\nSystem Statistics:")
        print(json.dumps(stats, indent=2))
    
    # Clear data
    if args.clear:
        system.clear_all_data()
        print("✓ All data cleared")
    
    # Answer question
    if args.question:
        print(f"\nQuestion: {args.question}")
        result = system.ask_question(args.question)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sources: {result['num_sources']}")
        
        if result['sources']:
            print("\nSource Documents:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']} - {source['section']} (Page {source['page']})")
                print(f"   Score: {source['score']:.3f}")
                print(f"   Preview: {source['content_preview']}")
                print()
    
    # Interactive mode
    if args.interactive:
        print("\nInteractive mode started. Type 'quit' to exit, 'stats' for statistics.")
        while True:
            try:
                question = input("\nAsk a medical question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'stats':
                    stats = system.get_system_stats()
                    print(json.dumps(stats, indent=2))
                    continue
                elif not question:
                    continue
                
                result = system.ask_question(question)
                
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.2f}")
                
                if result['sources']:
                    print(f"\nSources ({result['num_sources']}):")
                    for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                        print(f"{i}. {source['filename']} - {source['section']}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
