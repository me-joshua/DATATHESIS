def build_prompt(user_riddle, matches):
    """
    Constructs a research-grade prompt for Layer 3 (LLM Reasoning).
    Uses the standardized Title Case labels: Logic, Mathematical, Wordplay, Cultural.
    """
    
    # 1. Definitive Research Categories
    category_definitions = """
    - Logic: Functional deduction based on physical properties (e.g., An egg has no doors/windows).
    - Mathematical: Quantitative queries involving counting, calculations, or shares.
    - Wordplay: Linguistic puns, phonetic patterns, or literal play on words.
    - Cultural: Use of personification, South Indian metaphors (e.g., coconut as 'three eyes'), or regional deities/festivals.
    """

    # 2. Contextual Examples (Few-Shot Learning)
    examples_str = ""
    if not matches:
        examples_str = "No specific database matches. Use expert linguistic intuition."
    else:
        for i, m in enumerate(matches, 1):
            examples_str += f"Reference {i}:\n"
            examples_str += f"Riddle: {m['Question']}\n"
            examples_str += f"Answer: {m['Answer']}\n"
            examples_str += f"Category: {m['Category']}\n\n"

    # 3. The Final Prompt Construction
    prompt = f"""You are a Senior Linguistic Researcher specializing in South Indian (Tamil and Malayalam) folk literature.

### CORE TASK:
Classify the following riddle into exactly one of these four categories: [Logic, Mathematical, Wordplay, Cultural].

### CATEGORY DEFINITIONS:
{category_definitions}

### DATABASE CONTEXT:
The following examples from our dataset show similar semantic patterns:
{examples_str}

### TEST RIDDLE:
"{user_riddle}"

### INSTRUCTIONS:
1. Analyze if the riddle uses personification (Cultural) or raw physics (Logic).
2. Check for numerical requirements (Mathematical).
3. Output ONLY the category name. No periods, no extra words, no explanation.

Category:"""

    return prompt