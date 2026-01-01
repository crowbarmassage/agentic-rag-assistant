# ShopUNow FAQ Data Generator

> Generate synthetic FAQ data for the ShopUNow AI Assistant knowledge base

---

## 📋 Overview

This tool generates realistic FAQ (Question-Answer) pairs for ShopUNow's four departments using LLM APIs. The generated data populates the vector store used by the RAG pipeline.

### Departments

| Department | User Type | Description |
|------------|-----------|-------------|
| **HR** | Internal Employee | PTO, payroll, benefits, policies, onboarding |
| **IT Support** | Internal Employee | Passwords, VPN, hardware, software, email |
| **Billing** | External Customer | Refunds, invoices, payments, gift cards |
| **Shipping** | External Customer | Tracking, returns, delays, damaged goods |

### Output

Each department generates **15 QA pairs** (configurable), totaling **60 QA pairs** across all departments.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai  # Or: google-generativeai, groq
```

### 2. Set API Key

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or Gemini
export GOOGLE_API_KEY="AIza..."

# Or Groq
export GROQ_API_KEY="gsk_..."
```

### 3. Run Generator

```bash
python generate_faqs_standalone.py
```

### 4. Check Output

```
data/raw/
├── hr_faqs.json           # 15 QA pairs
├── it_support_faqs.json   # 15 QA pairs
├── billing_faqs.json      # 15 QA pairs
└── shipping_faqs.json     # 15 QA pairs
```

---

## ⚙️ CLI Options

```bash
python generate_faqs_standalone.py [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `openai` | LLM provider: `openai`, `gemini`, `groq` |
| `--model` | `-m` | (default) | Specific model name |
| `--num-pairs` | `-n` | `15` | QA pairs per department |
| `--output-dir` | `-o` | `./data/raw` | Output directory |
| `--departments` | `-d` | (all) | Specific departments to generate |

### Examples

```bash
# Use Gemini instead of OpenAI
python generate_faqs_standalone.py --provider gemini

# Use specific model
python generate_faqs_standalone.py --provider openai --model gpt-4o

# Generate 20 pairs per department
python generate_faqs_standalone.py --num-pairs 20

# Generate only HR and Billing
python generate_faqs_standalone.py --departments hr billing

# Custom output directory
python generate_faqs_standalone.py --output-dir ./my_data

# Combined options
python generate_faqs_standalone.py -p groq -n 20 -d hr it_support -o ./custom_data
```

---

## 🤖 Supported Providers

| Provider | Models | Default | Cost (60 QA pairs) |
|----------|--------|---------|-------------------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4-turbo | gpt-4o-mini | ~$0.10-0.20 |
| **Gemini** | gemini-1.5-flash, gemini-1.5-pro | gemini-1.5-flash | ~$0.05-0.10 |
| **Groq** | llama-3.3-70b-versatile, mixtral-8x7b | llama-3.3-70b | Free tier available |

---

## 📄 Output Format

Each JSON file contains:

```json
{
  "department": "hr",
  "department_name": "Human Resources",
  "user_type": "internal_employee",
  "description": "Human Resources department for ShopUNow...",
  "generated_at": "2025-01-01T12:00:00.000000",
  "count": 15,
  "qa_pairs": [
    {
      "id": "hr_001",
      "question": "How do I apply for paid time off (PTO)?",
      "answer": "You can apply for PTO through the HR Portal at hr.shopunow.com. Navigate to 'Time Off Requests', select your dates, and submit for manager approval. Requests should be submitted at least 2 weeks in advance for planned leave. You'll receive email confirmation once approved.",
      "department": "hr",
      "user_type": "internal_employee",
      "keywords": ["PTO", "time off", "leave", "vacation", "HR portal"]
    },
    {
      "id": "hr_002",
      "question": "When is payday?",
      "answer": "ShopUNow processes payroll bi-weekly on Fridays...",
      "department": "hr",
      "user_type": "internal_employee", 
      "keywords": ["payday", "payroll", "salary", "direct deposit"]
    }
  ]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique identifier: `{department}_{number}` |
| `question` | Realistic FAQ question |
| `answer` | Detailed, actionable answer (50-150 words) |
| `department` | Department key |
| `user_type` | `internal_employee` or `external_customer` |
| `keywords` | Search/filter keywords |

---

## 📁 File Structure

```
shopunow-ai-assistant/
├── generate_faqs_standalone.py    # ← Main script (self-contained)
├── requirements_datagen.txt       # Minimal dependencies
├── data/
│   └── raw/                       # Generated output
│       ├── hr_faqs.json
│       ├── it_support_faqs.json
│       ├── billing_faqs.json
│       └── shipping_faqs.json
└── src/
    └── utils/                     # Package version (alternative)
        ├── data_generation_prompts.py
        └── generate_faq_data.py
```

### Which Script to Use?

| Script | Use When |
|--------|----------|
| `generate_faqs_standalone.py` | Quick generation, no project setup needed |
| `src/utils/generate_faq_data.py` | Integrated with full project structure |

---

## 🎯 Customization

### Adding a New Department

Edit `DEPARTMENT_CONTEXTS` in the script:

```python
DEPARTMENT_CONTEXTS = {
    # ... existing departments ...
    
    "returns": {
        "name": "Returns & Exchanges",
        "user_type": "external_customer",
        "description": """Returns department for ShopUNow.
        
Handles:
- Return eligibility and policies
- Exchange requests
- Refund processing
- Store credit
- Damaged item replacements"""
    }
}
```

Then run:

```bash
python generate_faqs_standalone.py --departments returns
```

### Adjusting Answer Quality

Modify the prompt in `get_faq_generation_prompt()`:

```python
# For shorter answers
"Answers should be helpful and concise (30-50 words each)"

# For more detailed answers
"Answers should be comprehensive and thorough (100-200 words each)"

# For specific tone
"Answers should be warm, empathetic, and customer-focused"
```

---

## ⚠️ Troubleshooting

### API Key Not Found

```
Error: OpenAI API key not found
```

**Fix:** Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### JSON Parse Error

```
Error: Failed to parse JSON
```

**Cause:** LLM returned malformed JSON or extra text.

**Fix:** The script auto-retries with JSON extraction. If persistent, try:
- Different model (`--model gpt-4o`)
- Different provider (`--provider gemini`)

### Rate Limiting

```
Error: Rate limit exceeded
```

**Fix:** Wait and retry, or use a different provider:
```bash
python generate_faqs_standalone.py --provider groq
```

### Empty Keywords

Some LLMs may return empty keyword arrays. This is acceptable—keywords are optional for retrieval.

---

## 💰 Cost Estimates

| Provider | Model | 60 QA Pairs | 120 QA Pairs |
|----------|-------|-------------|--------------|
| OpenAI | gpt-4o-mini | ~$0.10 | ~$0.20 |
| OpenAI | gpt-4o | ~$1.00 | ~$2.00 |
| Gemini | gemini-1.5-flash | ~$0.05 | ~$0.10 |
| Gemini | gemini-1.5-pro | ~$0.50 | ~$1.00 |
| Groq | llama-3.3-70b | Free* | Free* |

*Groq has generous free tier limits

---

## 🔗 Next Steps

After generating FAQ data:

1. **Ingest into ChromaDB:**
   ```python
   from src.vectorstore import ingest_faqs
   ingest_faqs(reset_collection=True)
   ```

2. **Verify ingestion:**
   ```python
   from src.vectorstore import verify_ingestion
   verify_ingestion()
   ```

3. **Start the API:**
   ```bash
   python run.py
   ```

---

## 📝 Sample Output

### HR FAQ Sample

**Q:** How do I apply for paid time off (PTO)?

**A:** You can apply for PTO through the HR Portal at hr.shopunow.com. Navigate to 'Time Off Requests' in the left menu, select your desired dates, choose the leave type (vacation, sick, or personal), and submit for manager approval. Requests should be submitted at least 2 weeks in advance for planned leave. You'll receive an email notification once your manager approves or requests changes. Check your remaining PTO balance anytime on the portal dashboard.

---

### IT Support FAQ Sample

**Q:** How do I reset my password?

**A:** To reset your password, visit the IT Self-Service Portal at it.shopunow.com and click 'Forgot Password'. Enter your employee ID and work email, then check your inbox for a reset link (valid for 15 minutes). If you're locked out completely, call the IT Helpdesk at ext. 5555 (available 24/7) for immediate assistance. Remember: passwords must be 12+ characters with uppercase, lowercase, numbers, and symbols.

---

### Billing FAQ Sample

**Q:** How do I request a refund?

**A:** To request a refund, log into your account at my.shopunow.com and go to 'Order History'. Find the order, click 'Request Refund', select the items and reason, then submit. Refunds for credit card payments process within 5-7 business days; PayPal refunds appear within 3-5 days. You'll receive email confirmation with a reference number. For orders over $500 or special circumstances, our team reviews requests within 24-48 hours.

---

### Shipping FAQ Sample

**Q:** Where is my order?

**A:** Track your order at my.shopunow.com/track or use the tracking link in your shipping confirmation email. Enter your order number (starts with SUN-) to see real-time status including current location, estimated delivery, and delivery attempts. Standard shipping takes 5-7 business days; Express is 2-3 days. If tracking shows 'delivered' but you haven't received it, check with neighbors and building management, then contact us within 48 hours to file a claim.

---

<p align="center">
  <i>Part of the ShopUNow AI Assistant project</i>
</p>
