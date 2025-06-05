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


def template_net_profit_margin():
    """1:Basic: Net Profit Margin Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    revenue = random.randint(50000, 500000)  # Revenue
    net_income = random.randint(5000, 100000)  # Net Income

    question = (
        f"{company_name}, operating in the {industry} industry, reported a total revenue of ${revenue} and a net income of ${net_income}. "
        f"Calculate the company's net profit margin."
    )

    # Step 1: Calculate the net profit margin
    net_profit_margin = round((net_income / revenue) * 100, 2)

    solution = (
        f"Step 1: Calculate the net profit margin using the formula:\n"
        f"  Net Profit Margin = (Net Income / Revenue) × 100\n"
        f"                   = ({net_income} / {revenue}) × 100 = {net_profit_margin:.2f}%"
    )

    return question, solution


def template_return_on_equity():
    """2:Basic: Return on Equity (ROE) Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    net_income = random.randint(10000, 200000)  # Net Income
    shareholders_equity = random.randint(50000, 300000)  # Shareholders' Equity

    question = (
        f"{company_name}, a leading firm in the {industry} sector, reported a net income of ${net_income} and total shareholders' equity "
        f"of ${shareholders_equity}. Calculate the company's Return on Equity (ROE)."
    )

    # Step 1: Calculate the Return on Equity
    return_on_equity = round((net_income / shareholders_equity) * 100, 2)

    solution = (
        f"Step 1: Calculate the Return on Equity using the formula:\n"
        f"  ROE = (Net Income / Shareholders' Equity) × 100\n"
        f"      = ({net_income} / {shareholders_equity}) × 100 = {return_on_equity:.2f}%"
    )

    return question, solution


def template_combined_net_profit_and_roe():
    """3:Intermediate: Combined Net Profit and ROE Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    revenue = random.randint(100000, 500000)  # Revenue
    net_income = random.randint(10000, 100000)  # Net Income
    shareholders_equity = random.randint(50000, 300000)  # Shareholders' Equity

    question = (
        f"{company_name}, operating in the {industry} industry, reported ${revenue} in total revenue, ${net_income} in net income, and "
        f"shareholders' equity of ${shareholders_equity}. Calculate both the company's Net Profit Margin and Return on Equity (ROE)."
    )

    # Step 1: Calculate Net Profit Margin
    net_profit_margin = round((net_income / revenue) * 100, 2)

    # Step 2: Calculate Return on Equity
    return_on_equity = round((net_income / shareholders_equity) * 100, 2)

    solution = (
        f"Step 1: Calculate the Net Profit Margin:\n"
        f"  Net Profit Margin = (Net Income / Revenue) × 100\n"
        f"                   = ({net_income} / {revenue}) × 100 = {net_profit_margin:.2f}%\n\n"
        f"Step 2: Calculate the Return on Equity:\n"
        f"  ROE = (Net Income / Shareholders' Equity) × 100\n"
        f"      = ({net_income} / {shareholders_equity}) × 100 = {return_on_equity:.2f}%"
    )

    return question, solution


def template_net_profit_with_target():
    """4:Intermediate: Net Profit Target Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    revenue = random.randint(100000, 400000)  # Revenue
    net_income = random.randint(5000, 100000)  # Net Income
    target_margin = random.uniform(10, 30)  # Target Net Profit Margin (%)

    question = (
        f"{company_name}, a company in the {industry} industry, reported revenue of ${revenue} and net income of ${net_income}. "
        f"The company has set a target net profit margin of {target_margin:.2f}%. Calculate the company's current net profit margin "
        f"and determine the additional net income required to meet the target."
    )

    # Step 1: Calculate current net profit margin
    current_margin = round((net_income / revenue) * 100, 2)

    # Step 2: Calculate additional net income required to meet target margin
    required_net_income = round((target_margin / 100) * revenue, 2)
    additional_income_needed = round(required_net_income - net_income, 2)

    solution = (
        f"Step 1: Calculate the current net profit margin:\n"
        f"  Net Profit Margin = (Net Income / Revenue) × 100\n"
        f"                   = ({net_income} / {revenue}) × 100 = {current_margin:.2f}%\n\n"
        f"Step 2: Calculate the additional net income required to meet the target margin:\n"
        f"  Required Net Income = (Target Margin × Revenue) / 100\n"
        f"                     = ({target_margin:.2f} × {revenue}) / 100 = {required_net_income:.2f}\n"
        f"  Additional Income Needed = {required_net_income:.2f} - {net_income} = {additional_income_needed:.2f}"
    )

    return question, solution


def template_roe_increasing_equity():
    """5:Advanced: Analyzes ROE changes with increasing equity, involving
    three steps: current ROE calculation, new equity determination, and
    projected ROE calculation with consideration of equity dilution effects."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    net_income = random.randint(20000, 150000)  # Net Income
    current_equity = random.randint(100000, 500000)  # Current Shareholders' Equity
    equity_increase = round((random.uniform(10, 30)),2)  # Percentage increase in equity

    question = (
        f"{company_name}, a top firm in the {industry} industry, reported net income of ${net_income} and current shareholders' equity "
        f"of ${current_equity}. The company plans to increase its equity by {equity_increase:.2f}%. Calculate the current Return on Equity "
        f"and the expected ROE after the equity increase."
    )

    # Step 1: Calculate current ROE
    current_roe = round((net_income / current_equity) * 100, 2)

    # Step 2: Calculate expected new equity
    new_equity = round(current_equity * (1 + equity_increase / 100), 2)

    # Step 3: Calculate expected new ROE after equity increase
    new_roe = round((net_income / new_equity) * 100, 2)

    solution = (
        f"Step 1: Calculate the current ROE:\n"
        f"  ROE = (Net Income / Shareholders' Equity) × 100\n"
        f"      = ({net_income} / {current_equity}) × 100 = {current_roe:.2f}%\n\n"
        f"Step 2: Calculate the expected new equity:\n"
        f"  New Equity = Current Equity × (1 + Percentage Increase)\n"
        f"            = {current_equity} × (1 + {equity_increase / 100:.2f}) = {new_equity:.2f}\n\n"
        f"Step 3: Calculate the expected new ROE:\n"
        f"  New ROE = (Net Income / New Equity) × 100\n"
        f"         = ({net_income} / {new_equity}) × 100 = {new_roe:.2f}%"
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
        template_net_profit_margin,
        template_return_on_equity,
        template_combined_net_profit_and_roe,
        template_net_profit_with_target,
        template_roe_increasing_equity,
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
    output_file = "../../testset/financial_ratios/profitratio.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(
        f"Successfully generated {len(all_problems)} problems and saved to {output_file}"
    )


if __name__ == "__main__":
    main()
