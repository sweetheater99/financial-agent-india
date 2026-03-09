def test_system_prompt_has_v6_sections():
    from claude_intel import SYSTEM_PROMPT
    required = [
        "MARKET MICROSTRUCTURE", "VWAP", "OI changes",
        "EXPIRY DAY RULES", "GAP HANDLING", "TIME AWARENESS",
        "POSITION MANAGEMENT", "IV AWARENESS", "HARD CONSTRAINTS",
        "Never add to losing positions",
    ]
    for section in required:
        assert section in SYSTEM_PROMPT, f"Missing: {section}"

def test_system_prompt_no_old_references():
    from claude_intel import SYSTEM_PROMPT
    removed = ["15+ years", "pithy"]
    for term in removed:
        assert term not in SYSTEM_PROMPT, f"Old reference: {term}"
