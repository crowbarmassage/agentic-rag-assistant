"""Prompts and contexts for generating synthetic FAQ data."""

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


def get_faq_generation_prompt(department_key: str, num_pairs: int = 15) -> str:
    """
    Generate prompt for creating FAQ pairs for a department.
    
    Args:
        department_key: Department identifier (hr, it_support, billing, shipping)
        num_pairs: Number of QA pairs to generate
        
    Returns:
        Formatted prompt string
    """
    context = DEPARTMENT_CONTEXTS.get(department_key)
    if not context:
        raise ValueError(f"Unknown department: {department_key}. Valid: {list(DEPARTMENT_CONTEXTS.keys())}")
    
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
