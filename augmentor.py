def build_prompt(user_riddle, matches):
    """Constructs a high-precision prompt with category definitions and linguistic context."""
    
    # 1. Define the categories to resolve ambiguity (especially Logic vs. Cultural)
    category_definitions = """
    - Logic: Requires deductive reasoning or identifying a clever hidden solution to a situation.
    - Mathematical: Focuses on numerical relationships, counting, or arithmetic operations.
    - Wordplay: Relies on puns, double meanings, or the literal sounds of words.
    - Cultural: Uses regional metaphors, personification of household items (like a traditional lamp or coconut), or specific South Indian traditions.
    """

    # 2. Format the retrieved matches
    examples_str = ""
    for i, m in enumerate(matches, 1):
        examples_str += f"Example {i}:\n"
        examples_str += f"Riddle: {m['Question']}\n"
        examples_str += f"Category: {m['Category']}\n\n"

    # 3. Enhanced Instruction Prompt
    prompt = f"""You are an expert South Indian riddle classifier specializing in Tamil and Malayalam linguistics.

### Classification Criteria:
{category_definitions}

### Reference Context (Top 3 similar riddles from database):
{examples_str}

### Task:
Analyze the new riddle below. Using the definitions and the examples provided, classify it into the most appropriate category. 

New Riddle: {user_riddle}

Output ONLY the category name (Logic, Mathematical, Wordplay, or Cultural). No extra text.

Category:"""

    return prompt