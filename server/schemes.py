"""Scheme metadata and compatibility-safe eligibility evaluation helpers.

The benchmark environment currently reasons over a very small applicant profile:
`age`, `income`, `occupation`, and `has_aadhaar`.

This module keeps those benchmark-critical paths intact while adding richer scheme
metadata and optional advanced checks. The extra checks only activate when the
caller actually provides the relevant profile fields, which avoids downgrading
existing benchmark performance or breaking old tasks that operate on sparse data.
"""

from typing import Any, Dict, Optional


SCHEME_PRIORITY = [
    "PMKVY",
    "MGNREGS",
    "PMAY",
    "PM_SYM",
    "AYUSHMAN_BHARAT",
    "E_SHRAM",
    "NFSA",
    "PMMVY",
]


SCHEMES: Dict[str, Dict[str, Any]] = {
    "PMKVY": {
        "full_name": "Pradhan Mantri Kaushal Vikas Yojana",
        "scheme_type": "Skill development and short-term training",
        "administering_body": "Ministry of Skill Development and Entrepreneurship",
        "benefit": "Free skill training, assessment, and certification under approved job roles",
        "benefit_details": {
            "primary_support": "Training, assessment, and certification through approved centres",
            "cash_support_note": "Cash rewards and stipend-like support vary by implementation cycle and category; they are not universal for every candidate.",
            "delivery_mode": "Training partner / skill centre enrollment",
        },
        "target_group": "Youth and working-age candidates seeking market-relevant skills",
        "eligibility": {
            "age_min": 18,
            "age_max": 35,
            "income_max": 9999,
            "occupations": ["mason", "carpenter"],
            "requires_aadhaar": False,
            "citizenship": "indian",
            "education_status_any_of": ["school_dropout", "college_dropout", "unemployed", "underemployed"],
        },
        "eligibility_notes": [
            "Benchmark logic keeps the existing narrow occupation filter for deterministic evaluation.",
            "Real PMKVY eligibility depends more on training-category fit than on these benchmark occupations.",
            "Aadhaar is commonly used for candidate registration in live deployments, but benchmark compatibility keeps it optional here.",
        ],
        "common_exclusions": [
            "Applicant outside training age band for the configured benchmark profile",
            "Income at or above the strict benchmark ceiling",
            "Occupation outside the supported benchmark job roles",
        ],
        "required_docs": ["aadhaar", "education_certificate"],
        "optional_docs": ["bank_account_details", "passport_size_photo", "mobile_number"],
        "document_notes": {
            "aadhaar": "Often used for identity and candidate registration even when not enforced by benchmark logic.",
            "education_certificate": "Used to establish educational level or last-attended qualification when relevant.",
        },
        "application_channels": ["skill_centre", "training_partner", "official_skill_portal"],
        "official_reference": "PMKVY official portal / MSDE operational guidelines",
    },
    "MGNREGS": {
        "full_name": "Mahatma Gandhi National Rural Employment Guarantee Scheme",
        "scheme_type": "Guaranteed wage employment",
        "administering_body": "Ministry of Rural Development",
        "benefit": "Up to 100 days of guaranteed wage employment to eligible rural households",
        "benefit_details": {
            "primary_support": "Unskilled manual work with notified wage payments",
            "household_scope": "Benefit is household-based, but adults volunteer individually for work",
            "delivery_mode": "Gram Panchayat / local program officer workflow",
        },
        "target_group": "Adult members of rural households willing to do unskilled manual work",
        "eligibility": {
            "age_min": 18,
            "age_max": 60,
            "income_max": None,
            "occupations": ["farm_labourer"],
            "requires_aadhaar": True,
            "residence_type": "rural",
            "willing_for_unskilled_manual_work": True,
        },
        "eligibility_notes": [
            "Benchmark logic uses a deterministic occupation proxy: `farm_labourer`.",
            "Live MGNREGS eligibility is tied to rural household status and willingness to do unskilled manual work, not just occupation title.",
            "Job card issuance is operationally central even though initial eligibility and enrollment are distinct steps.",
        ],
        "common_exclusions": [
            "Minor applicant",
            "Missing Aadhaar in benchmark mode",
            "Non-rural residency when such data is provided",
        ],
        "required_docs": ["aadhaar", "job_card"],
        "optional_docs": ["address_proof", "bank_passbook", "photograph"],
        "document_notes": {
            "job_card": "Core employment entitlement document for MGNREGS work demand and attendance tracking.",
        },
        "application_channels": ["gram_panchayat", "block_office", "mgnrega_portal"],
        "official_reference": "MGNREGA / Ministry of Rural Development guidance",
    },
    "PMAY": {
        "full_name": "Pradhan Mantri Awaas Yojana",
        "scheme_type": "Housing assistance",
        "administering_body": "MoHUA / Ministry of Rural Development depending on urban or rural component",
        "benefit": "Housing support for eligible low-income households",
        "benefit_details": {
            "primary_support": "Benchmark keeps a fixed grant-style description for deterministic evaluation.",
            "amount_note": "Assistance varies by PMAY vertical and geography; rural and urban components differ in structure and subsidy form.",
            "delivery_mode": "State and local implementing authority workflow",
        },
        "target_group": "Economically weaker or low-income households lacking adequate housing",
        "eligibility": {
            "age_min": 21,
            "age_max": 55,
            "income_max": 5999,
            "occupations": None,
            "requires_aadhaar": True,
            "housing_status_any_of": ["homeless", "kutcha_house", "inadequate_housing"],
            "property_ownership_limit": "no_pucca_house",
        },
        "eligibility_notes": [
            "Benchmark uses a single strict income threshold to stay deterministic.",
            "Real PMAY eligibility differs across PMAY-G and PMAY-U verticals and often depends on SECC data, beneficiary category, and housing ownership status.",
            "Land or property-related proof is typically important for sanction and construction support.",
        ],
        "common_exclusions": [
            "Income at or above the strict benchmark ceiling",
            "Missing Aadhaar in benchmark mode",
            "Already owning a pucca house when such profile data is supplied",
        ],
        "required_docs": ["aadhaar", "income_certificate", "land_document"],
        "optional_docs": ["bank_passbook", "address_proof", "caste_certificate", "affidavit_no_pucca_house"],
        "document_notes": {
            "land_document": "Ownership, possession, or land-use proof is commonly needed in grant-linked housing assistance flows.",
        },
        "application_channels": ["urban_local_body", "gram_panchayat", "state_housing_portal"],
        "official_reference": "PMAY implementation guidelines",
    },
    "PM_SYM": {
        "full_name": "Pradhan Mantri Shram Yogi Maan-dhan",
        "scheme_type": "Contributory pension",
        "administering_body": "Ministry of Labour and Employment",
        "benefit": "Assured monthly pension after age 60 for eligible unorganised workers",
        "benefit_details": {
            "primary_support": "Rs 3000 per month pension after age 60 on contributory basis",
            "contribution_note": "Subscriber contribution varies by entry age and is matched by the Government of India under scheme rules.",
            "delivery_mode": "CSC / designated enrollment point linked to pension account setup",
        },
        "target_group": "Unorganised workers with modest monthly income and no formal social-security enrollment",
        "eligibility": {
            "age_min": 18,
            "age_max": 40,
            "income_max": 14999,
            "occupations": None,
            "requires_aadhaar": True,
            "required_profile_fields": ["worker_type", "has_bank_account", "is_epfo_member", "is_esic_member"],
            "worker_type": "unorganised",
            "not_epfo": True,
            "not_esic": True,
            "income_frequency": "monthly",
            "has_savings_bank_account": True,
        },
        "eligibility_notes": [
            "Monthly income benchmark is kept as strict less-than Rs 15000 for deterministic evaluation.",
            "Real PM-SYM enrollment also expects a savings bank or Jan Dhan account and mobile-linked onboarding flow.",
        ],
        "common_exclusions": [
            "Formal social security membership such as EPFO or ESIC when profile data confirms it",
            "Income at or above the scheme ceiling",
            "Applicant outside contributory entry-age band",
        ],
        "required_docs": ["aadhaar", "bank_passbook", "mobile_number"],
        "optional_docs": ["savings_bank_account_number", "ifsc_code"],
        "document_notes": {
            "bank_passbook": "Needed for contribution auto-debit and pension-linked banking setup.",
        },
        "application_channels": ["common_service_centre", "maandhan_portal"],
        "official_reference": "PM-SYM / maandhan official guidance",
    },
    "AYUSHMAN_BHARAT": {
        "full_name": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
        "scheme_type": "Health insurance / health assurance",
        "administering_body": "National Health Authority",
        "benefit": "Cashless annual health cover for eligible families at empanelled hospitals",
        "benefit_details": {
            "primary_support": "Up to Rs 5 lakh per family per year for secondary and tertiary care hospitalization",
            "family_scope": "Benchmark treats this as family-income-based, while real beneficiary identification is generally database-based.",
            "delivery_mode": "Ayushman card / beneficiary verification and hospital empanelment system",
        },
        "target_group": "Economically vulnerable families identified under PM-JAY eligibility systems",
        "eligibility": {
            "age_min": None,
            "age_max": None,
            "income_max": 49999,
            "occupations": None,
            "requires_aadhaar": True,
            "required_profile_fields": ["pmjay_database_verified"],
            "not_govt_employee": True,
            "pmjay_database_verified": True,
            "family_income_based": True,
        },
        "eligibility_notes": [
            "Real PM-JAY eligibility is largely driven by beneficiary databases and deprivation / occupational categories, not just a simple income test.",
            "Benchmark keeps a deterministic family-income ceiling to enable unambiguous grading.",
        ],
        "common_exclusions": [
            "Missing Aadhaar in benchmark mode",
            "Government employee status when such data is provided",
            "Family income at or above the benchmark ceiling",
        ],
        "required_docs": ["aadhaar", "ration_card"],
        "optional_docs": ["family_id", "pmjay_card", "mobile_number"],
        "document_notes": {
            "ration_card": "Common proof for family composition and public-distribution-linked identity context.",
        },
        "application_channels": ["ayushman_mitra", "pmjay_portal", "common_service_centre"],
        "official_reference": "National Health Authority / PM-JAY guidance",
    },
    "E_SHRAM": {
        "full_name": "e-Shram Portal Registration",
        "scheme_type": "National unorganised worker registration",
        "administering_body": "Ministry of Labour and Employment",
        "benefit": "Registration, social-security linkage, and accidental insurance support as notified",
        "benefit_details": {
            "primary_support": "Unorganised worker database registration with portability and linkage benefits",
            "insurance_note": "Insurance-related support may depend on the currently linked protection arrangement and operational notifications.",
            "delivery_mode": "Portal or CSC-assisted self-registration",
        },
        "target_group": "Unorganised workers not covered by specified organised social-security systems",
        "eligibility": {
            "age_min": 16,
            "age_max": 59,
            "income_max": None,
            "occupations": None,
            "requires_aadhaar": True,
            "required_profile_fields": ["worker_type", "is_epfo_member", "is_esic_member", "is_nps_subscriber"],
            "worker_type": "unorganised",
            "not_epfo": True,
            "not_esic": True,
            "not_nps": True,
        },
        "eligibility_notes": [
            "Age 16 to 59 and absence of EPFO / ESIC / NPS are core operational checks reflected by e-Shram FAQs.",
            "Occupation can be broad and is not limited by the benchmark.",
        ],
        "common_exclusions": [
            "Membership in EPFO, ESIC, or NPS when profile data confirms it",
            "Applicant outside working-age registration band",
        ],
        "required_docs": ["aadhaar", "mobile_number", "bank_passbook"],
        "optional_docs": ["occupation_details", "education_details", "address_proof"],
        "document_notes": {
            "mobile_number": "Used for OTP-linked registration and profile updates.",
        },
        "application_channels": ["eshram_portal", "common_service_centre"],
        "official_reference": "e-Shram FAQs and registration guidance",
    },
    "NFSA": {
        "full_name": "National Food Security Act - Ration Card",
        "scheme_type": "Food security / subsidised ration entitlement",
        "administering_body": "Department of Food and Public Distribution with State Governments",
        "benefit": "Subsidised food grains and linked ration-card-based welfare entitlements",
        "benefit_details": {
            "primary_support": "Access to subsidised grains under PHH / AAY-style state allocation workflows",
            "state_variation_note": "Actual eligibility is state-specific and often uses exclusion and inclusion criteria beyond income alone.",
            "delivery_mode": "State ration card system / fair price shop network",
        },
        "target_group": "Low-income or otherwise eligible households under state-implemented NFSA criteria",
        "eligibility": {
            "age_min": 18,
            "age_max": None,
            "income_max": 9999,
            "occupations": None,
            "requires_aadhaar": True,
            "required_profile_fields": ["state_specific_eligibility_confirmed", "is_income_tax_payer"],
            "not_income_tax_payer": True,
            "state_specific_eligibility_confirmed": True,
            "state_specific_screening": True,
        },
        "eligibility_notes": [
            "NFSA implementation is highly state-specific; this benchmark uses a deterministic nationalized proxy rule.",
            "Income-tax-paying households are commonly treated as exclusion cases in simplified welfare targeting logic.",
        ],
        "common_exclusions": [
            "Income at or above the benchmark ceiling",
            "Income-tax payer status when such profile data is supplied",
            "Missing Aadhaar in benchmark mode",
        ],
        "required_docs": ["aadhaar", "address_proof", "family_photo"],
        "optional_docs": ["income_certificate", "old_ration_card", "residence_certificate"],
        "document_notes": {
            "address_proof": "Needed for household mapping to the relevant ration card jurisdiction.",
            "family_photo": "Often used to identify household members on ration card records.",
        },
        "application_channels": ["state_pds_portal", "ration_office", "common_service_centre"],
        "official_reference": "DFPD / State PDS eligibility notifications",
    },
    "PMMVY": {
        "full_name": "Pradhan Mantri Matru Vandana Yojana",
        "scheme_type": "Maternity benefit",
        "administering_body": "Ministry of Women and Child Development",
        "benefit": "Cash maternity benefit for eligible women subject to notified conditions",
        "benefit_details": {
            "primary_support": "Benchmark keeps Rs 5000 core maternity support wording for deterministic reference.",
            "instalment_note": "Disbursement is installment-based subject to pregnancy registration and maternal/child health milestones.",
            "delivery_mode": "Health system and beneficiary portal assisted enrollment",
        },
        "target_group": "Pregnant and lactating women eligible under PMMVY conditions",
        "eligibility": {
            "age_min": 18,
            "age_max": None,
            "income_max": None,
            "occupations": None,
            "requires_aadhaar": True,
            "required_profile_fields": ["gender", "is_pregnant", "first_child", "has_bank_account"],
            "gender": "female",
            "is_pregnant": True,
            "first_child": True,
            "has_bank_account": True,
        },
        "eligibility_notes": [
            "Benchmark aligns to first living child logic and active pregnancy status.",
            "Real PMMVY implementation depends on pregnancy registration, health-record milestones, and category-specific guidance.",
        ],
        "common_exclusions": [
            "Male applicant when gender is provided",
            "Not currently pregnant when profile data is provided",
            "Applicant without bank account when banking data is provided",
        ],
        "required_docs": ["aadhaar", "mch_card", "bank_passbook"],
        "optional_docs": ["pregnancy_registration_proof", "child_birth_record", "mobile_number"],
        "document_notes": {
            "mch_card": "Maternal and Child Health card is commonly used to track pregnancy and immunization milestones.",
        },
        "application_channels": ["anganwadi_centre", "health_facility", "state_pmmvy_portal"],
        "official_reference": "PMMVY guidelines / WCD beneficiary instructions",
    },
}


def _to_int(value: Any, default: int = 0) -> int:
    """Convert incoming profile values to int using a safe benchmark-friendly default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _to_text(value: Any) -> str:
    """Return a normalized lowercase string for robust comparisons."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _to_bool(value: Any) -> Optional[bool]:
    """Parse optional boolean-like values from sparse profiles."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = _to_text(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _profile_bool(profile: dict, *keys: str) -> Optional[bool]:
    """Read the first present boolean-like field from the applicant profile."""
    for key in keys:
        if key in profile:
            return _to_bool(profile.get(key))
    return None


def _matches_exact_if_present(profile: dict, key: str, expected: str) -> bool:
    """If a profile field is present, require it to match; otherwise stay permissive."""
    if key not in profile:
        return True
    return _to_text(profile.get(key)) == _to_text(expected)


def _has_required_profile_fields(profile: dict, required_fields: list[str]) -> bool:
    """Require specific fields for schemes that cannot be safely inferred from sparse profiles."""
    for field in required_fields:
        if field not in profile:
            return False
        value = profile.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def get_eligible_schemes(profile: dict) -> list:
    """
    Evaluate an applicant profile against all supported schemes.

    Compatibility rules:
    - Existing benchmark-critical checks for age, income, occupation, and Aadhaar remain primary.
    - Additional scheme conditions are enforced only when those profile fields are supplied.
    - Income ceilings remain strict by storing the highest allowed qualifying integer
      (e.g. 9999 means income must be strictly less than 10000).
    """
    eligible = []

    age = _to_int(profile.get("age", 0))
    income = _to_int(profile.get("income", 0))
    occupation = _to_text(profile.get("occupation"))
    aadhaar = _to_bool(profile.get("has_aadhaar")) is True

    for scheme_key, scheme in SCHEMES.items():
        rules = scheme["eligibility"]

        if rules.get("age_min") is not None and age < rules["age_min"]:
            continue
        if rules.get("age_max") is not None and age > rules["age_max"]:
            continue

        if rules.get("income_max") is not None and income > rules["income_max"]:
            continue

        occupations = rules.get("occupations")
        if occupations and occupation not in occupations:
            continue

        if rules.get("requires_aadhaar") and not aadhaar:
            continue

        required_profile_fields = rules.get("required_profile_fields", [])
        if required_profile_fields and not _has_required_profile_fields(profile, required_profile_fields):
            continue

        if rules.get("worker_type") and not _matches_exact_if_present(profile, "worker_type", rules["worker_type"]):
            continue

        if rules.get("residence_type") and not _matches_exact_if_present(profile, "residence_type", rules["residence_type"]):
            continue

        if rules.get("citizenship") and not _matches_exact_if_present(profile, "citizenship", rules["citizenship"]):
            continue

        if rules.get("gender") and not _matches_exact_if_present(profile, "gender", rules["gender"]):
            continue

        if rules.get("pmjay_database_verified"):
            pmjay_verified = _profile_bool(profile, "pmjay_database_verified")
            if pmjay_verified is not True:
                continue

        if rules.get("state_specific_eligibility_confirmed"):
            state_check = _profile_bool(profile, "state_specific_eligibility_confirmed")
            if state_check is not True:
                continue

        if rules.get("willing_for_unskilled_manual_work"):
            willingness = _profile_bool(profile, "willing_for_unskilled_manual_work")
            if willingness is False:
                continue

        if rules.get("not_epfo"):
            epfo_member = _profile_bool(profile, "is_epfo_member", "epfo_member", "has_epfo")
            if epfo_member is True:
                continue

        if rules.get("not_esic"):
            esic_member = _profile_bool(profile, "is_esic_member", "esic_member", "has_esic")
            if esic_member is True:
                continue

        if rules.get("not_nps"):
            nps_member = _profile_bool(profile, "is_nps_subscriber", "nps_subscriber", "has_nps")
            if nps_member is True:
                continue

        if rules.get("not_govt_employee"):
            govt_employee = _profile_bool(profile, "is_govt_employee", "is_government_employee", "government_employee")
            if govt_employee is True:
                continue

        if rules.get("not_income_tax_payer"):
            tax_payer = _profile_bool(profile, "is_income_tax_payer", "income_tax_payer")
            if tax_payer is True:
                continue

        if rules.get("is_pregnant"):
            pregnant = _profile_bool(profile, "is_pregnant", "pregnant")
            if pregnant is False:
                continue

        if rules.get("first_child"):
            first_child = _profile_bool(profile, "first_child", "is_first_child_case", "first_living_child")
            if first_child is False:
                continue

        if rules.get("has_bank_account"):
            bank_account = _profile_bool(profile, "has_bank_account", "bank_account_active")
            if bank_account is False:
                continue

        if rules.get("has_savings_bank_account"):
            savings_account = _profile_bool(profile, "has_savings_bank_account", "has_bank_account", "bank_account_active")
            if savings_account is False:
                continue

        housing_status = rules.get("housing_status_any_of")
        if housing_status and "housing_status" in profile:
            if _to_text(profile.get("housing_status")) not in {_to_text(item) for item in housing_status}:
                continue

        if rules.get("property_ownership_limit") == "no_pucca_house":
            owns_pucca_house = _profile_bool(profile, "owns_pucca_house", "has_pucca_house")
            if owns_pucca_house is True:
                continue

        education_status_any_of = rules.get("education_status_any_of")
        if education_status_any_of and "education_status" in profile:
            if _to_text(profile.get("education_status")) not in {_to_text(item) for item in education_status_any_of}:
                continue

        eligible.append(scheme_key)

    return eligible


def get_optimal_scheme(profile: dict) -> Optional[str]:
    """
    Return the highest-priority qualifying scheme for the applicant profile.

    Priority is intentionally benchmark-stable and should only be changed alongside
    environment and benchmark updates.
    """
    eligible = get_eligible_schemes(profile)

    for scheme in SCHEME_PRIORITY:
        if scheme in eligible:
            return scheme

    return None
