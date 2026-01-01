#!/usr/bin/env python3
"""
Standalone script to generate synthetic FAQ data for ShopUNow departments.
No package dependencies - can be run directly.

Usage:
    python generate_faqs_standalone.py
    python generate_faqs_standalone.py --provider openai --model gpt-4o-mini
    python generate_faqs_standalone.py --provider gemini
    python generate_faqs_standalone.py --departments hr billing

Environment Variables Required (depending on provider):
    OPENAI_API_KEY - for OpenAI provider
    GOOGLE_API_KEY - for Gemini provider  
    GROQ_API_KEY - for Groq provider
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Callable

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# =============================================================================
# DEPARTMENT CONTEXTS
# =============================================================================

DEPARTMENT_CONTEXTS = {
    "hr": {
        "name": "Human Resources",
        "user_type": "internal_employee",
        "description": """Human Resources department for ShopUNow, a retail company selling clothing, DIY products, books, and toys.

HR handles employee lifecycle matters including:
- Paid time off (PTO) and leave requests (vacation, sick, personal)
- Payroll questions and salary matters
- Benefits enrollment and inquiries (health, dental, 401k)
- Performance reviews and feedback processes
- Company policies and employee handbook
- Onboarding for new employees
- Internal job transfers and promotions
- Employee wellness programs
- Training and development opportunities
- Workplace conduct and HR policies
- Employee ID badges and access
- Employment verification requests"""
    },
    "it_support": {
        "name": "IT Support / Tech Support",
        "user_type": "internal_employee",
        "description": """IT Support department for ShopUNow, a retail company.

IT Support manages technical issues for employees including:
- Password resets and account lockouts
- VPN setup and remote access issues
- Hardware issues (laptops, monitors, keyboards, mice)
- Software installation and licensing requests
- Email and Outlook calendar problems
- Network and WiFi connectivity issues
- Security awareness and phishing reporting
- System outages and maintenance notifications
- New employee tech setup and equipment
- Printer and scanner issues
- Multi-factor authentication (MFA) setup
- Shared drive and file access permissions"""
    },
    "billing": {
        "name": "Billing & Payments",
        "user_type": "external_customer",
        "description": """Billing & Payments department for ShopUNow, a retail company.

Billing handles customer payment matters including:
- Invoice requests and copies
- Refund processing and status inquiries
- Payment method updates (credit card, PayPal, etc.)
- Overcharge disputes and billing errors
- Promotional code and discount issues
- Gift card balance and redemption
- Payment declined/failed troubleshooting
- Tax exemption requests
- Receipt and purchase history requests
- Store credit inquiries
- Subscription and recurring billing
- Price match requests"""
    },
    "shipping": {
        "name": "Shipping & Delivery",
        "user_type": "external_customer",
        "description": """Shipping & Delivery department for ShopUNow, a retail company.

Shipping handles order fulfillment queries including:
- Order tracking and delivery status
- Estimated delivery time inquiries
- Delayed or late package issues
- Missing or lost package claims
- Damaged goods reporting and replacement
- Return initiation and return labels
- Address changes before shipping
- International shipping questions
- In-store pickup options
- Delivery scheduling and rescheduling
- Shipping cost questions
- Package theft claims"""
    }
}


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def get_faq_generation_prompt(department_key: str, num_pairs: int = 15) -> str:
    """Generate prompt for creating FAQ pairs for a department."""
    context = DEPARTMENT_CONTEXTS[department_key]
    user_type_desc = "internal employees of the company" if context["user_type"] == "internal_employee" else "external customers"
    
    return f"""Act as an expert in running the {context['name']} department for ShopUNow, a retail company selling clothing, DIY products, books, and toys.

{context['description']}

Create a list of {num_pairs} Questions and Answers which could be the most frequently asked questions for the {context['name']} department, typically asked by {user_type_desc}.

Requirements:
1. Questions should be realistic and diverse, covering different aspects of the department
2. Answers should be helpful, specific, and actionable (50-150 words each)
3. Include a mix of simple and complex questions
4. Answers should reference realistic systems, portals, or processes (e.g., "HR Portal at hr.shopunow.com", "IT Helpdesk at ext. 5555", "customer portal at my.shopunow.com")
5. Make answers sound professional but friendly
6. Include specific timeframes where relevant (e.g., "within 3-5 business days", "processed on the last working day of each month")

Return the response as a JSON array with this exact structure:
[
  {{
    "question": "The question text here",
    "answer": "The detailed answer here with specific information and next steps",
    "keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  ...
]

Generate exactly {num_pairs} QA pairs. Output ONLY the JSON array, no additional text or markdown formatting."""


# =============================================================================
# LLM PROVIDERS
# =============================================================================

def create_openai_generator(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """Create OpenAI generation function."""
    from openai import OpenAI
    client = OpenAI()
    
    def generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    return generate


def create_gemini_generator(model: str = "gemini-1.5-flash") -> Callable[[str], str]:
    """Create Gemini generation function."""
    import google.generativeai as genai
    genai_model = genai.GenerativeModel(model)
    
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
    
    return generate


def create_groq_generator(model: str = "llama-3.3-70b-versatile") -> Callable[[str], str]:
    """Create Groq generation function."""
    from groq import Groq
    client = Groq()
    
    def generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    return generate


def create_generator(provider: str, model: Optional[str] = None) -> tuple[Callable[[str], str], str]:
    """
    Create generator function for specified provider.
    
    Returns:
        Tuple of (generate_function, model_name)
    """
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "groq": "llama-3.3-70b-versatile"
    }
    
    model_name = model or defaults.get(provider, "unknown")
    
    if provider == "openai":
        return create_openai_generator(model_name), model_name
    elif provider == "gemini":
        return create_gemini_generator(model_name), model_name
    elif provider == "groq":
        return create_groq_generator(model_name), model_name
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, gemini, groq")


# =============================================================================
# DATA GENERATION
# =============================================================================

def parse_llm_response(raw_response: str) -> list[dict]:
    """Parse LLM JSON response, handling various formats."""
    try:
        data = json.loads(raw_response)
        
        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common wrapper keys
            for key in ["qa_pairs", "faqs", "questions", "data", "items"]:
                if key in data:
                    return data[key]
            # Single key wrapper
            if len(data) == 1:
                return list(data.values())[0]
        
        return data
        
    except json.JSONDecodeError as e:
        # Try to extract JSON array from response
        import re
        match = re.search(r'\[[\s\S]*\]', raw_response)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {raw_response[:500]}")


def generate_department_faqs(
    department_key: str,
    generate_fn: Callable[[str], str],
    num_pairs: int = 15
) -> dict:
    """Generate FAQ data for a single department."""
    context = DEPARTMENT_CONTEXTS[department_key]
    prompt = get_faq_generation_prompt(department_key, num_pairs)
    
    print(f"  Generating {num_pairs} FAQs for {context['name']}...")
    
    # Call LLM
    raw_response = generate_fn(prompt)
    
    # Parse response
    qa_list = parse_llm_response(raw_response)
    
    # Build QA pairs with metadata
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(qa_pairs),
        "qa_pairs": qa_pairs
    }


def generate_all_faqs(
    provider: str = "openai",
    model: Optional[str] = None,
    num_pairs: int = 15,
    output_dir: str = "./data/raw",
    departments: Optional[list] = None
) -> dict:
    """Generate FAQ data for all (or specified) departments."""
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    print(f"\n{'='*60}")
    print("ShopUNow FAQ Data Generator")
    print(f"{'='*60}")
    
    generate_fn, model_name = create_generator(provider, model)
    
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Output: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Determine departments
    dept_keys = departments or list(DEPARTMENT_CONTEXTS.keys())
    
    results = {}
    total_pairs = 0
    
    for dept_key in dept_keys:
        if dept_key not in DEPARTMENT_CONTEXTS:
            print(f"WARNING: Unknown department '{dept_key}', skipping")
            continue
        
        try:
            # Generate data
            data = generate_department_faqs(dept_key, generate_fn, num_pairs)
            
            # Save to file
            output_file = output_path / f"{dept_key}_faqs.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to {output_file}")
            
            results[dept_key] = data
            total_pairs += data["count"]
            
        except Exception as e:
            print(f"  ✗ ERROR generating {dept_key}: {e}")
            raise
    
    # Print summary
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"Total QA pairs: {total_pairs}")
    
    for dept_key, data in results.items():
        icon = "👤" if data["user_type"] == "internal_employee" else "🛒"
        print(f"  {icon} {dept_key}: {data['count']} pairs")
    
    print(f"\nFiles saved to: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic FAQ data for ShopUNow departments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_faqs_standalone.py
  python generate_faqs_standalone.py --provider openai --model gpt-4o
  python generate_faqs_standalone.py --provider gemini
  python generate_faqs_standalone.py --provider groq --num-pairs 20
  python generate_faqs_standalone.py --departments hr billing

Environment Variables:
  OPENAI_API_KEY  - Required for OpenAI provider
  GOOGLE_API_KEY  - Required for Gemini provider
  GROQ_API_KEY    - Required for Groq provider
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "gemini", "groq"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (uses provider default if not specified)"
    )
    
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        default=15,
        help="QA pairs per department (default: 15)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/raw",
        help="Output directory (default: ./data/raw)"
    )
    
    parser.add_argument(
        "--departments", "-d",
        nargs="+",
        choices=list(DEPARTMENT_CONTEXTS.keys()),
        default=None,
        help="Specific departments (default: all)"
    )
    
    args = parser.parse_args()
    
    generate_all_faqs(
        provider=args.provider,
        model=args.model,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        departments=args.departments
    )


if __name__ == "__main__":
    main()
