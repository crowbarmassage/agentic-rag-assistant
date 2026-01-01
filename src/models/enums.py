"""Enumeration types for ShopUNow Assistant."""

from enum import Enum


class Department(str, Enum):
    """ShopUNow department identifiers."""
    HR = "hr"
    IT_SUPPORT = "it_support"
    BILLING = "billing"
    SHIPPING = "shipping"
    UNKNOWN = "unknown"

    @classmethod
    def internal_departments(cls) -> list["Department"]:
        """Departments serving internal employees."""
        return [cls.HR, cls.IT_SUPPORT]

    @classmethod
    def external_departments(cls) -> list["Department"]:
        """Departments serving external customers."""
        return [cls.BILLING, cls.SHIPPING]

    @classmethod
    def valid_departments(cls) -> list["Department"]:
        """All valid departments (excludes UNKNOWN)."""
        return [cls.HR, cls.IT_SUPPORT, cls.BILLING, cls.SHIPPING]

    @classmethod
    def get_description(cls, dept: "Department") -> str:
        """Get human-readable description for department."""
        descriptions = {
            cls.HR: "Human Resources - employee lifecycle, leave, payroll, benefits, policies",
            cls.IT_SUPPORT: "IT Support - technical issues, hardware, software, system access",
            cls.BILLING: "Billing & Payments - invoices, refunds, payment methods, overcharges",
            cls.SHIPPING: "Shipping & Delivery - order tracking, delays, returns, damaged goods",
            cls.UNKNOWN: "Unknown department"
        }
        return descriptions.get(dept, "Unknown")


class UserType(str, Enum):
    """User classification."""
    INTERNAL_EMPLOYEE = "internal_employee"
    EXTERNAL_CUSTOMER = "external_customer"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    """Query sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class RouteDecision(str, Enum):
    """Routing decision outcomes."""
    RAG_PIPELINE = "rag_pipeline"
    HUMAN_ESCALATION = "human_escalation"
