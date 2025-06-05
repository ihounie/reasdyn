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


def template_asset_turnover_simple():
    """1:Basic: Asset Turnover Ratio Calculation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate sales and assets and round them to 2 decimal places
    total_sales = round(random.uniform(50000, 500000), 2)  # Total sales
    total_assets = round(random.uniform(100000, 1000000), 2)  # Total assets

    question = (
        f"{company_name}, operating in the {industry} industry, has generated total sales of ${total_sales} and has total assets of ${total_assets}. "
        f"Calculate the company's Asset Turnover Ratio."
    )

    # Calculate the Asset Turnover Ratio using the rounded values
    asset_turnover = round(total_sales / total_assets, 2)

    solution = (
        f"Step 1: Calculate the Asset Turnover Ratio using the formula:\n"
        f"  Asset Turnover Ratio = Total Sales / Total Assets\n"
        f"                     = ${total_sales} / ${total_assets} = {asset_turnover:.2f}"
    )

    return question, solution


def template_asset_turnover_increase_assets():
    """2:Basic: Asset Turnover with Asset Increase"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round sales, assets, and asset increase to 2 decimal places
    total_sales = round(random.uniform(100000, 1000000), 2)  # Total sales
    total_assets_last_year = round(
        random.uniform(200000, 2000000), 2
    )  # Assets last year
    asset_increase = round(random.uniform(10000, 200000), 2)  # Increase in assets
    total_assets_this_year = round(total_assets_last_year + asset_increase, 2)

    question = (
        f"{company_name}, a major player in the {industry} industry, generated sales of ${total_sales} last year and had total assets of ${total_assets_last_year}. "
        f"This year, the company's assets increased by ${asset_increase}, while its sales remained constant. How did the change in assets impact the company's Asset Turnover Ratio?"
    )

    # Step 1: Calculate last year's Asset Turnover Ratio using the rounded values
    asset_turnover_last_year = round(total_sales / total_assets_last_year, 2)

    # Step 2: Calculate this year's Asset Turnover Ratio using the rounded values
    asset_turnover_this_year = round(total_sales / total_assets_this_year, 2)

    solution = (
        f"Step 1: Calculate last year's Asset Turnover Ratio:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets_last_year} = {asset_turnover_last_year:.2f}\n\n"
        f"Step 2: Calculate this year's Asset Turnover Ratio:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets_this_year} = {asset_turnover_this_year:.2f}"
    )

    return question, solution


def template_asset_turnover_with_depreciation():
    """3:Intermediate: Asset Turnover with Depreciation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round sales, assets, and depreciation to 2 decimal places
    total_sales = round(random.uniform(200000, 1500000), 2)  # Total sales
    total_assets = round(random.uniform(500000, 3000000), 2)  # Total assets
    depreciation = round(random.uniform(50000, 500000), 2)  # Depreciation on assets
    adjusted_assets = round(total_assets - depreciation, 2)

    question = (
        f"{company_name}, operating in the {industry} sector, generated ${total_sales} in sales this year and had total assets of ${total_assets}. "
        f"However, due to asset depreciation of ${depreciation}, the company's total assets have reduced by that amount. "
        f"Calculate the company's Asset Turnover Ratio after accounting for depreciation."
    )

    # Step 1: Calculate Asset Turnover Ratio before depreciation using rounded values
    asset_turnover_before = round(total_sales / total_assets, 2)

    # Step 2: Calculate Asset Turnover Ratio after depreciation using adjusted total assets
    asset_turnover_after = round(total_sales / adjusted_assets, 2)

    solution = (
        f"Step 1: Calculate Asset Turnover Ratio before depreciation:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets} = {asset_turnover_before:.2f}\n\n"
        f"Step 2: Adjust total assets for depreciation:\n"
        f"  Adjusted Assets = ${total_assets} - ${depreciation} = ${adjusted_assets}\n\n"
        f"Step 3: Calculate Asset Turnover Ratio after depreciation:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${adjusted_assets} = {asset_turnover_after:.2f}"
    )

    return question, solution


def template_asset_turnover_scenario_analysis():
    """4:Intermediate: Asset Turnover Scenario Analysis"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round sales, assets, and increase values to 2 decimal places
    total_sales = round(random.uniform(400000, 2000000), 2)  # Total sales
    total_assets = round(random.uniform(500000, 2500000), 2)  # Total assets
    asset_increase = round(
        random.uniform(50000, 400000), 2
    )  # Asset increase for Scenario 1
    sales_increase_percentage = round(
        random.uniform(10, 50), 2
    )  # Sales increase in Scenario 2
    new_sales = round(
        total_sales * (1 + sales_increase_percentage / 100), 2
    )  # New sales after increase

    question = (
        f"{company_name}, in the {industry} industry, had total sales of ${total_sales} and total assets of ${total_assets}. The company is evaluating two options:\n"
        f"- Scenario 1: Increase its assets by ${asset_increase} to improve production capacity.\n"
        f"- Scenario 2: Focus on maximizing sales, which could increase sales by {sales_increase_percentage}%. "
        f"For each scenario, how will the company's Asset Turnover Ratio change?"
    )

    # Step 1: Calculate current Asset Turnover Ratio
    asset_turnover_current = round(total_sales / total_assets, 2)

    # Step 2: Calculate Asset Turnover Ratio for Scenario 1 (increased assets)
    new_assets_scenario_1 = round(total_assets + asset_increase, 2)
    asset_turnover_scenario_1 = round(total_sales / new_assets_scenario_1, 2)

    # Step 3: Calculate Asset Turnover Ratio for Scenario 2 (increased sales)
    asset_turnover_scenario_2 = round(new_sales / total_assets, 2)

    solution = (
        f"Step 1: Calculate the current Asset Turnover Ratio:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets} = {asset_turnover_current:.2f}\n\n"
        f"Step 2: Calculate the Asset Turnover Ratio for Scenario 1 (increased assets):\n"
        f"  New Assets = ${total_assets} + ${asset_increase} = ${new_assets_scenario_1}\n"
        f"  Asset Turnover Ratio (Scenario 1) = ${total_sales} / ${new_assets_scenario_1} = {asset_turnover_scenario_1:.2f}\n\n"
        f"Step 3: Calculate the Asset Turnover Ratio for Scenario 2 (increased sales):\n"
        f"  New Sales = ${new_sales}\n"
        f"  Asset Turnover Ratio (Scenario 2) = ${new_sales} / ${total_assets} = {asset_turnover_scenario_2:.2f}"
    )

    return question, solution


def template_asset_turnover_investment_depreciation():
    """5:Advanced: Asset Turnover with Investment and Depreciation"""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)

    # Generate and round sales, assets, investment, and depreciation to 2 decimal places
    total_sales = round(random.uniform(500000, 2500000), 2)  # Total sales
    total_assets = round(random.uniform(800000, 3000000), 2)  # Total assets
    new_investment = round(
        random.uniform(100000, 500000), 2
    )  # New investment in assets
    depreciation = round(random.uniform(50000, 200000), 2)  # Depreciation on assets

    question = (
        f"{company_name}, a leader in the {industry} industry, generated total sales of ${total_sales} and had total assets of ${total_assets}. "
        f"This year, the company plans to invest ${new_investment} in new machinery, while also experiencing asset depreciation of ${depreciation}. "
        f"How will the company's Asset Turnover Ratio change after accounting for the new investment and depreciation?"
    )

    # Step 1: Calculate current Asset Turnover Ratio
    asset_turnover_before = round(total_sales / total_assets, 2)

    # Step 2: Adjust total assets for new investment and depreciation
    adjusted_assets = round(total_assets + new_investment - depreciation, 2)

    # Step 3: Calculate Asset Turnover Ratio after adjustments
    asset_turnover_after = round(total_sales / adjusted_assets, 2)

    solution = (
        f"Step 1: Calculate the current Asset Turnover Ratio:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets} = {asset_turnover_before:.2f}\n\n"
        f"Step 2: Adjust total assets for new investment and depreciation:\n"
        f"  Adjusted Assets = ${total_assets} + ${new_investment} - ${depreciation} = ${adjusted_assets}\n\n"
        f"Step 3: Calculate Asset Turnover Ratio after adjustments:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${adjusted_assets} = {asset_turnover_after:.2f}"
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
        template_asset_turnover_simple,
        template_asset_turnover_increase_assets,
        template_asset_turnover_with_depreciation,
        template_asset_turnover_scenario_analysis,
        template_asset_turnover_investment_depreciation,
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
    output_file = "../../testset/financial_ratios/effratio.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(
        f"Successfully generated {len(all_problems)} problems and saved to {output_file}"
    )


if __name__ == "__main__":
    main()
