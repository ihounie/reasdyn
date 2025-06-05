import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = [
    "automotive",
    "technology",
    "e-commerce",
    "aerospace",
    "internet services",
]


def template_current_ratio_simple():
    """1:Basic: Current Ratio Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    current_assets = random.randint(5000, 50000)  # Current Assets
    current_liabilities = random.randint(2000, 25000)  # Current Liabilities

    question = (
        f"{company_name}, operating in the {industry} industry, has current assets amounting to ${current_assets} "
        f"and current liabilities of ${current_liabilities}. Calculate the current ratio."
    )

    # Step 1: Calculate the current ratio with consistent precision throughout
    current_ratio = round(current_assets / current_liabilities, 2)

    solution = (
        f"Step 1: Calculate the current ratio using the formula:\n"
        f"  Current Ratio = Current Assets / Current Liabilities\n"
        f"               = {current_assets} / {current_liabilities} = {current_ratio:.2f}"
    )

    return question, solution


def template_quick_ratio():
    """2:Basic: Quick Ratio Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    current_assets = random.randint(10000, 100000)  # Current Assets
    inventories = random.randint(2000, 30000)  # Inventories
    current_liabilities = random.randint(5000, 50000)  # Current Liabilities

    question = (
        f"{company_name}, a major player in the {industry} industry, has ${current_assets} in current assets, ${inventories} in inventories, "
        f"and ${current_liabilities} in current liabilities. Calculate the quick ratio (acid-test ratio)."
    )

    # Step 1: Calculate the quick ratio with consistent precision
    quick_ratio = round((current_assets - inventories) / current_liabilities, 2)

    solution = (
        f"Step 1: Calculate the quick ratio using the formula:\n"
        f"  Quick Ratio = (Current Assets - Inventories) / Current Liabilities\n"
        f"             = ({current_assets} - {inventories}) / {current_liabilities} = {quick_ratio:.2f}"
    )

    return question, solution


def template_both_ratios():
    """3:Intermediate: Current and Quick Ratio Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    current_assets = random.randint(20000, 200000)  # Current Assets
    inventories = random.randint(5000, 50000)  # Inventories
    prepaid_expenses = random.randint(1000, 20000)  # Prepaid Expenses
    current_liabilities = random.randint(10000, 100000)  # Current Liabilities

    question = (
        f"{company_name}, operating in the {industry} sector, has ${current_assets} in current assets, ${inventories} in inventories, "
        f"${prepaid_expenses} in prepaid expenses, and ${current_liabilities} in current liabilities. Calculate both the current ratio and quick ratio."
    )

    # Step 1: Calculate the current ratio with consistent precision
    current_ratio = round(current_assets / current_liabilities, 2)

    # Step 2: Calculate the quick ratio with consistent precision
    quick_ratio = round(
        (current_assets - inventories - prepaid_expenses) / current_liabilities, 2
    )

    solution = (
        f"Step 1: Calculate the current ratio:\n"
        f"  Current Ratio = Current Assets / Current Liabilities\n"
        f"               = {current_assets} / {current_liabilities} = {current_ratio:.2f}\n\n"
        f"Step 2: Calculate the quick ratio:\n"
        f"  Quick Ratio = (Current Assets - Inventories - Prepaid Expenses) / Current Liabilities\n"
        f"             = ({current_assets} - {inventories} - {prepaid_expenses}) / {current_liabilities} = {quick_ratio:.2f}"
    )

    return question, solution


def template_quick_ratio_scenario():
    """4:Intermediate: Quick Ratio Scenario Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    current_assets = random.randint(50000, 500000)  # Current Assets
    inventories = random.randint(10000, 100000)  # Inventories
    prepaid_expenses = random.randint(2000, 25000)  # Prepaid Expenses
    current_liabilities = random.randint(25000, 250000)  # Current Liabilities
    min_quick_ratio = round(random.uniform(1.5, 2.5), 2)  # Minimum Quick Ratio required

    question = (
        f"{company_name}, in the {industry} industry, has ${current_assets} in current assets, ${inventories} in inventories, "
        f"${prepaid_expenses} in prepaid expenses, and ${current_liabilities} in current liabilities. "
        f"To meet liquidity requirements, the company must maintain a minimum quick ratio of {min_quick_ratio:.2f}. "
        f"How much additional cash or marketable securities are needed to meet this requirement?"
    )

    # Step 1: Calculate the existing quick ratio with consistent precision
    quick_ratio = round(
        (current_assets - inventories - prepaid_expenses) / current_liabilities, 2
    )

    # Step 2: Calculate the required quick assets to meet the minimum quick ratio
    required_quick_assets = round(min_quick_ratio * current_liabilities, 2)

    # Step 3: Calculate the additional quick assets needed
    additional_assets_needed = round(
        required_quick_assets - (current_assets - inventories - prepaid_expenses), 2
    )

    solution = (
        f"Step 1: Calculate the existing quick ratio:\n"
        f"  Quick Ratio = (Current Assets - Inventories - Prepaid Expenses) / Current Liabilities\n"
        f"             = ({current_assets} - {inventories} - {prepaid_expenses}) / {current_liabilities} = {quick_ratio:.2f}\n\n"
        f"Step 2: Calculate the required quick assets to meet the minimum quick ratio:\n"
        f"  Required Quick Assets = Minimum Quick Ratio × Current Liabilities\n"
        f"                       = {min_quick_ratio:.2f} × {current_liabilities} = {required_quick_assets:.2f}\n\n"
        f"Step 3: Calculate the additional quick assets needed:\n"
        f"  Additional Quick Assets Needed = Required Quick Assets - Existing Quick Assets\n"
        f"                               = {required_quick_assets:.2f} - ({current_assets} - {inventories} - {prepaid_expenses}) = {additional_assets_needed:.2f}"
    )

    return question, solution


def template_liquidity_minimum_quick_ratio():
    """5:Advanced: Comprehensive Liquidity Analysis"""
    # Defining variables with ranges for the company's balance sheet items
    A = random.randint(25000, 200000)  # Cash
    B = random.randint(5000, 50000)  # Marketable securities
    C = random.randint(20000, 150000)  # Accounts receivable
    D = random.randint(15000, 100000)  # Inventory
    E = random.randint(1000, 10000)  # Prepaid expenses
    F = random.randint(30000, 300000)  # Current liabilities
    G = round(random.uniform(1.5, 2.5), 2)  # Minimum Quick Ratio

    question = (
        f"A company's balance sheet provides the following details:\n"
        f"  - Cash: ${A}\n"
        f"  - Marketable securities: ${B}\n"
        f"  - Accounts receivable: ${C}\n"
        f"  - Inventory: ${D}\n"
        f"  - Prepaid expenses: ${E}\n"
        f"  - Current liabilities: ${F}\n\n"
        f"The company is evaluating its liquidity health under two different scenarios:\n"
        f"1. Calculate the Current Ratio and Quick Ratio.\n"
        f"2. Assume the company needs to maintain a minimum Quick Ratio of {G}. "
        f"How much additional cash or marketable securities are required for the company to meet this threshold?"
    )

    # Step 1: Compute Current Ratio
    current_assets = A + B + C + D + E
    current_ratio = round(current_assets / F, 2)

    # Step 2: Compute Quick Ratio
    quick_assets = A + B + C  # Quick assets exclude Inventory and Prepaid Expenses
    quick_ratio = round(quick_assets / F, 2)

    # Step 3: Calculate the additional cash or marketable securities needed to meet the minimum Quick Ratio
    required_quick_assets = round(G * F, 2)  # To meet the minimum Quick Ratio
    additional_assets_needed = max(0, round(required_quick_assets - quick_assets, 2))

    solution = (
        f"Step 1: Calculate the Current Ratio:\n"
        f"  Current Assets = Cash + Marketable Securities + Accounts Receivable + Inventory + Prepaid Expenses\n"
        f"                 = ${A} + ${B} + ${C} + ${D} + ${E} = ${current_assets}\n"
        f"  Current Ratio = Current Assets / Current Liabilities = ${current_assets} / ${F} = {current_ratio:.2f}\n\n"
        f"Step 2: Calculate the Quick Ratio:\n"
        f"  Quick Assets = Cash + Marketable Securities + Accounts Receivable\n"
        f"              = ${A} + ${B} + ${C} = ${quick_assets}\n"
        f"  Quick Ratio = Quick Assets / Current Liabilities = ${quick_assets} / ${F} = {quick_ratio:.2f}\n\n"
        f"Step 3: Determine how much additional cash or marketable securities are required:\n"
        f"  Required Quick Assets = Minimum Quick Ratio × Current Liabilities\n"
        f"                       = {G} × ${F} = ${required_quick_assets}\n"
        f"  Additional cash or marketable securities needed = ${required_quick_assets} - ${quick_assets} = ${additional_assets_needed:.2f}"
    )

    return question, solution


def main():
    """
    Generate 10 instances of each template with different random seeds
    and write the results to a JSON file.
    """
    import json

    # List of template functions
    templates = [
        template_current_ratio_simple,
        template_quick_ratio,
        template_both_ratios,
        template_quick_ratio_scenario,
        template_liquidity_minimum_quick_ratio,
    ]

    # List to store all generated problems
    all_problems = []

    # Generate 10 problems for each template
    for template_func in templates:
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()

        for i in range(10):
            # Generate a unique seed for each problem
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)

            # Generate the problem and solution
            question, solution = template_func()

            # Create a JSON entry
            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution,
            }

            # Add to the list of problems
            all_problems.append(problem_entry)

            # Reset the random seed
            random.seed()

    random.shuffle(all_problems)
    # Write all problems to a .jsonl file
    output_file = "../../testset/financial_ratios/liqratio.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(
        f"Successfully generated {len(all_problems)} problems and saved to {output_file}"
    )


if __name__ == "__main__":
    main()
