"""
Script to generate synthetic FAQ data for all departments.

Usage:
    python generate_faq_data.py
    
    # Or with options:
    python generate_faq_data.py --provider openai --model gpt-4o-mini --num-pairs 15
    
Environment Variables Required:
    OPENAI_API_KEY - for OpenAI provider
    GOOGLE_API_KEY - for Gemini provider  
    GROQ_API_KEY - for Groq provider
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path for imports when running as script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_generation_prompts import get_faq_generation_prompt, DEPARTMENT_CONTEXTS


def create_llm_client(provider: str, model: Optional[str] = None):
    """
    Create LLM client based on provider.
    
    Args:
        provider: One of 'openai', 'gemini', 'groq'
        model: Specific model name (uses default if None)
        
    Returns:
        Tuple of (client, model_name, generate_function)
    """
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model_name = model or "gpt-4o-mini"
        
        def generate(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        
        return client, model_name, generate
    
    elif provider == "gemini":
        import google.generativeai as genai
        model_name = model or "gemini-1.5-flash"
        genai_model = genai.GenerativeModel(model_name)
        
        def generate(prompt: str) -> str:
            response = genai_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                    response_mime_type="application/json"
                )
            )
            return response.text
        
        return genai_model, model_name, generate
    
    elif provider == "groq":
        from groq import Groq
        client = Groq()
        model_name = model or "llama-3.3-70b-versatile"
        
        def generate(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        
        return client, model_name, generate
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, gemini, groq")


def generate_department_faqs(
    department_key: str,
    generate_fn,
    num_pairs: int = 15
) -> dict:
    """
    Generate FAQ data for a single department.
    
    Args:
        department_key: Department identifier
        generate_fn: Function to call LLM
        num_pairs: Number of QA pairs to generate
        
    Returns:
        Dictionary with department FAQ data
    """
    context = DEPARTMENT_CONTEXTS[department_key]
    prompt = get_faq_generation_prompt(department_key, num_pairs)
    
    print(f"  Generating {num_pairs} FAQs for {context['name']}...")
    
    # Call LLM
    raw_response = generate_fn(prompt)
    
    # Parse JSON - handle potential wrapper
    try:
        data = json.loads(raw_response)
        # Handle if response is wrapped in an object
        if isinstance(data, dict) and "qa_pairs" in data:
            qa_list = data["qa_pairs"]
        elif isinstance(data, dict) and len(data) == 1:
            # Single key wrapper
            qa_list = list(data.values())[0]
        elif isinstance(data, list):
            qa_list = data
        else:
            qa_list = data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON: {e}")
        print(f"  Raw response (first 500 chars): {raw_response[:500]}")
        raise
    
    # Build QA pairs with IDs
    qa_pairs = []
    for i, item in enumerate(qa_list):
        qa_pairs.append({
            "id": f"{department_key}_{i+1:03d}",
            "question": item["question"],
            "answer": item["answer"],
            "department": department_key,
            "user_type": context["user_type"],
            "keywords": item.get("keywords", [])
        })
    
    print(f"  ✓ Generated {len(qa_pairs)} QA pairs")
    
    return {
        "department": department_key,
        "department_name": context["name"],
        "user_type": context["user_type"],
        "description": context["description"],
        "generated_at": datetime.utcnow().isoformat(),
        "count": len(qa_pairs),
        "qa_pairs": qa_pairs
    }


def save_department_data(data: dict, output_dir: Path) -> Path:
    """Save department FAQ data to JSON file."""
    output_file = output_dir / f"{data['department']}_faqs.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to {output_file}")
    return output_file


def generate_all_faqs(
    provider: str = "openai",
    model: Optional[str] = None,
    num_pairs: int = 15,
    output_dir: str = "./data/raw",
    departments: Optional[list] = None
) -> dict:
    """
    Generate FAQ data for all (or specified) departments.
    
    Args:
        provider: LLM provider name
        model: Specific model (uses default if None)
        num_pairs: Number of QA pairs per department
        output_dir: Directory to save JSON files
        departments: List of department keys (all if None)
        
    Returns:
        Dictionary with generation summary
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create LLM client
    print(f"\n{'='*50}")
    print("ShopUNow FAQ Data Generator")
    print(f"{'='*50}")
    
    _, model_name, generate_fn = create_llm_client(provider, model)
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Output: {output_path.absolute()}")
    print(f"{'='*50}\n")
    
    # Determine departments to process
    dept_keys = departments or list(DEPARTMENT_CONTEXTS.keys())
    
    results = {}
    total_pairs = 0
    
    for dept_key in dept_keys:
        if dept_key not in DEPARTMENT_CONTEXTS:
            print(f"WARNING: Unknown department '{dept_key}', skipping")
            continue
        
        try:
            data = generate_department_faqs(dept_key, generate_fn, num_pairs)
            save_department_data(data, output_path)
            results[dept_key] = data
            total_pairs += data["count"]
        except Exception as e:
            print(f"  ERROR generating {dept_key}: {e}")
            raise
    
    # Print summary
    print(f"\n{'='*50}")
    print("Generation Complete!")
    print(f"{'='*50}")
    print(f"Total QA pairs: {total_pairs}")
    for dept_key, data in results.items():
        user_type = "👤 Employee" if data["user_type"] == "internal_employee" else "🛒 Customer"
        print(f"  {dept_key}: {data['count']} pairs ({user_type})")
    print(f"\nFiles saved to: {output_path.absolute()}")
    
    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic FAQ data for ShopUNow departments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_faq_data.py
  python generate_faq_data.py --provider openai --model gpt-4o-mini
  python generate_faq_data.py --provider gemini --num-pairs 20
  python generate_faq_data.py --departments hr it_support
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "gemini", "groq"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Specific model name (uses provider default if not specified)"
    )
    
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        default=15,
        help="Number of QA pairs per department (default: 15)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/raw",
        help="Output directory for JSON files (default: ./data/raw)"
    )
    
    parser.add_argument(
        "--departments", "-d",
        nargs="+",
        choices=list(DEPARTMENT_CONTEXTS.keys()),
        default=None,
        help="Specific departments to generate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Run generation
    generate_all_faqs(
        provider=args.provider,
        model=args.model,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        departments=args.departments
    )


if __name__ == "__main__":
    main()
