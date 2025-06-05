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


def template_debt_to_equity_simple():
    """1:Basic: Debt-to-Equity Ratio Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round liabilities and equity to 2 decimal places for accuracy
    total_liabilities = round(random.uniform(10000, 50000), 2)  # Total liabilities
    shareholders_equity = round(random.uniform(5000, 25000), 2)  # Shareholders' equity

    question = (
        f"{company_name}, operating in the {industry} industry, has total liabilities of ${total_liabilities} and shareholders' equity of ${shareholders_equity}. "
        f"Calculate the company's Debt-to-Equity ratio."
    )

    # Step 1: Calculate the Debt-to-Equity ratio
    debt_to_equity = round(total_liabilities / shareholders_equity, 2)

    solution = (
        f"Step 1: Calculate the Debt-to-Equity ratio using the formula:\n"
        f"  Debt-to-Equity Ratio = Total Liabilities / Shareholders' Equity\n"
        f"                     = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity:.2f}"
    )

    return question, solution


def template_debt_to_equity_convertible_debt():
    """2:Basic: Debt-to-Equity with Convertible Debt"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round financial values to ensure consistency
    total_liabilities = round(random.uniform(30000, 100000), 2)  # Total liabilities
    convertible_debt = round(random.uniform(5000, 25000), 2)  # Convertible debt
    shareholders_equity = round(random.uniform(10000, 50000), 2)  # Shareholders' equity

    question = (
        f"{company_name}, a key player in the {industry} industry, has ${total_liabilities} in total liabilities, including ${convertible_debt} in convertible debt, "
        f"and ${shareholders_equity} in shareholders' equity. If the convertible debt converts into equity next year, how will the company's Debt-to-Equity ratio change?"
    )

    # Step 1: Calculate current Debt-to-Equity ratio
    debt_to_equity_before = round(total_liabilities / shareholders_equity, 2)

    # Step 2: Calculate new Debt-to-Equity ratio after debt converts to equity
    new_liabilities = round(total_liabilities - convertible_debt, 2)
    new_equity = round(shareholders_equity + convertible_debt, 2)
    debt_to_equity_after = round(new_liabilities / new_equity, 2)

    solution = (
        f"Step 1: Calculate the current Debt-to-Equity ratio:\n"
        f"  Debt-to-Equity Ratio = Total Liabilities / Shareholders' Equity\n"
        f"                     = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity_before:.2f}\n\n"
        f"Step 2: Calculate the new Debt-to-Equity ratio after debt conversion:\n"
        f"  New Liabilities = Total Liabilities - Convertible Debt = ${total_liabilities} - ${convertible_debt} = ${new_liabilities}\n"
        f"  New Shareholders' Equity = Shareholders' Equity + Convertible Debt = ${shareholders_equity} + ${convertible_debt} = ${new_equity}\n"
        f"  New Debt-to-Equity Ratio = ${new_liabilities} / ${new_equity} = {debt_to_equity_after:.2f}"
    )

    return question, solution


def template_debt_to_equity_capital_injection():
    """3:Intermediate: Debt-to-Equity with Capital Injection"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round financial values to ensure consistency
    total_liabilities = round(random.uniform(100000, 500000), 2)  # Total liabilities
    shareholders_equity = round(
        random.uniform(50000, 300000), 2
    )  # Shareholders' equity
    new_equity_injection = round(
        random.uniform(10000, 100000), 2
    )  # New equity injection

    question = (
        f"{company_name}, a prominent company in the {industry} industry, has ${total_liabilities} in total liabilities and ${shareholders_equity} in shareholders' equity. "
        f"The company plans to raise an additional ${new_equity_injection} in equity capital next quarter. How will the Debt-to-Equity ratio change after the capital injection?"
    )

    # Step 1: Calculate current Debt-to-Equity ratio
    debt_to_equity_before = round(total_liabilities / shareholders_equity, 2)

    # Step 2: Calculate new shareholders' equity after the capital injection
    new_equity = round(shareholders_equity + new_equity_injection, 2)

    # Step 3: Calculate new Debt-to-Equity ratio
    debt_to_equity_after = round(total_liabilities / new_equity, 2)

    solution = (
        f"Step 1: Calculate the current Debt-to-Equity ratio:\n"
        f"  Debt-to-Equity Ratio = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity_before:.2f}\n\n"
        f"Step 2: Calculate the new shareholders' equity after the capital injection:\n"
        f"  New Shareholders' Equity = ${shareholders_equity} + ${new_equity_injection} = ${new_equity}\n\n"
        f"Step 3: Calculate the new Debt-to-Equity ratio:\n"
        f"  New Debt-to-Equity Ratio = ${total_liabilities} / ${new_equity} = {debt_to_equity_after:.2f}"
    )

    return question, solution


def template_debt_to_equity_debt_repayment():
    """4:Intermediate: Debt-to-Equity with Debt Changes"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round financial values for consistency
    total_liabilities = round(random.uniform(200000, 1000000), 2)  # Total liabilities
    shareholders_equity = round(
        random.uniform(100000, 500000), 2
    )  # Shareholders' equity
    debt_repayment = round(random.uniform(50000, 200000), 2)  # Debt repayment
    new_borrowing = round(random.uniform(50000, 300000), 2)  # New short-term borrowing

    question = (
        f"{company_name}, operating in the {industry} sector, has ${total_liabilities} in total liabilities and ${shareholders_equity} in shareholders' equity. "
        f"It repays ${debt_repayment} of its long-term debt and borrows an additional ${new_borrowing} in short-term loans. "
        f"How will these changes affect the company's Debt-to-Equity ratio?"
    )

    # Step 1: Calculate the current Debt-to-Equity ratio
    debt_to_equity_before = round(total_liabilities / shareholders_equity, 2)

    # Step 2: Calculate new liabilities after debt repayment and new borrowing
    new_liabilities = round(total_liabilities - debt_repayment + new_borrowing, 2)

    # Step 3: Recalculate the Debt-to-Equity ratio after changes
    debt_to_equity_after = round(new_liabilities / shareholders_equity, 2)

    solution = (
        f"Step 1: Calculate the current Debt-to-Equity ratio:\n"
        f"  Debt-to-Equity Ratio = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity_before:.2f}\n\n"
        f"Step 2: Calculate new liabilities after debt repayment and new borrowing:\n"
        f"  New Liabilities = ${total_liabilities} - ${debt_repayment} + ${new_borrowing} = ${new_liabilities}\n\n"
        f"Step 3: Calculate the new Debt-to-Equity ratio:\n"
        f"  New Debt-to-Equity Ratio = ${new_liabilities} / ${shareholders_equity} = {debt_to_equity_after:.2f}"
    )

    return question, solution


def template_debt_to_equity_scenario_analysis():
    """5:Advanced: Debt-to-Equity Scenario Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round financial values for consistency
    total_liabilities = round(random.uniform(300000, 1500000), 2)  # Total liabilities
    shareholders_equity = round(
        random.uniform(200000, 1000000), 2
    )  # Shareholders' equity
    new_debt_issue = round(
        random.uniform(50000, 500000), 2
    )  # New debt to be issued in Scenario 1
    new_equity_issue = round(
        random.uniform(50000, 500000), 2
    )  # New equity to be issued in Scenario 2

    question = (
        f"{company_name}, operating in the {industry} industry, has ${total_liabilities} in total liabilities and ${shareholders_equity} in shareholders' equity. It is considering two scenarios:\n"
        f"1. Issue ${new_debt_issue} in new debt.\n"
        f"2. Issue ${new_equity_issue} in new equity.\n"
        f"For each scenario, how will the company's Debt-to-Equity ratio change?"
    )

    # Step 1: Calculate the current Debt-to-Equity ratio
    debt_to_equity_before = round(total_liabilities / shareholders_equity, 2)

    # Step 2: Calculate the Debt-to-Equity ratio for Scenario 1 (new debt)
    new_liabilities_scenario_1 = round(total_liabilities + new_debt_issue, 2)
    debt_to_equity_scenario_1 = round(
        new_liabilities_scenario_1 / shareholders_equity, 2
    )

    # Step 3: Calculate the Debt-to-Equity ratio for Scenario 2 (new equity)
    new_equity_scenario_2 = round(shareholders_equity + new_equity_issue, 2)
    debt_to_equity_scenario_2 = round(total_liabilities / new_equity_scenario_2, 2)

    solution = (
        f"Step 1: Calculate the current Debt-to-Equity ratio:\n"
        f"  Debt-to-Equity Ratio = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity_before:.2f}\n\n"
        f"Step 2: Calculate the Debt-to-Equity ratio for Scenario 1 (new debt issued):\n"
        f"  New Liabilities = ${total_liabilities} + ${new_debt_issue} = ${new_liabilities_scenario_1}\n"
        f"  Debt-to-Equity Ratio (Scenario 1) = ${new_liabilities_scenario_1} / ${shareholders_equity} = {debt_to_equity_scenario_1:.2f}\n\n"
        f"Step 3: Calculate the Debt-to-Equity ratio for Scenario 2 (new equity issued):\n"
        f"  New Shareholders' Equity = ${shareholders_equity} + ${new_equity_issue} = ${new_equity_scenario_2}\n"
        f"  Debt-to-Equity Ratio (Scenario 2) = ${total_liabilities} / ${new_equity_scenario_2} = {debt_to_equity_scenario_2:.2f}"
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
        template_debt_to_equity_simple,
        template_debt_to_equity_convertible_debt,
        template_debt_to_equity_capital_injection,
        template_debt_to_equity_debt_repayment,
        template_debt_to_equity_scenario_analysis,
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
    output_file = "../../testset/financial_ratios/levratio.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(
        f"Successfully generated {len(all_problems)} problems and saved to {output_file}"
    )


if __name__ == "__main__":
    main()
