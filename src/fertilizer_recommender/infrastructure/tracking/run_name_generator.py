from datetime import datetime

def generate_run_name(prefix: str = "CATB_Run") -> str:
    """
    Generate a unique MLflow run name with timestamp.

    Format:
    <PREFIX>_YYYY-MM-DD_HHhMMmSSs

    Example:
    XGB_run_2025-12-20_16h42m18s
    """
    if not prefix or not prefix.strip():
        raise ValueError("Le préfixe du run MLflow ne peut pas être vide.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    return f"{prefix}_{timestamp}"
