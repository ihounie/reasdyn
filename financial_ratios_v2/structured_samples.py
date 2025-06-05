import random
import json
import re
import os

# Named entities for companies and industries
company_names = ["Company A"]
industry_names = [
    "automotive",
    "technology",
    "e-commerce",
    "aerospace",
    "internet services",
]


def template_asset_turnover_investment_depreciation_structured():
    """Structured version of Asset Turnover with Investment and Depreciation template"""
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
    
    # Step 4: Calculate the change in Asset Turnover Ratio
    asset_turnover_change = round(asset_turnover_after - asset_turnover_before, 2)

    solution = (
        f"Step 1: Calculate the current Asset Turnover Ratio:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${total_assets} = {asset_turnover_before:.2f}\n\n"
        f"Step 2: Adjust total assets for new investment and depreciation:\n"
        f"  Adjusted Assets = ${total_assets} + ${new_investment} - ${depreciation} = ${adjusted_assets}\n\n"
        f"Step 3: Calculate Asset Turnover Ratio after adjustments:\n"
        f"  Asset Turnover Ratio = ${total_sales} / ${adjusted_assets} = {asset_turnover_after:.2f}\n\n"
        f"Step 4: Calculate the change in Asset Turnover Ratio:\n"
        f"  Change in Asset Turnover Ratio = {asset_turnover_after:.2f} - {asset_turnover_before:.2f} = {asset_turnover_change:.2f}"
    )

    # Create the structured output
    structured_output = {
        "question": question,
        "original_answer": solution,
        "final_result": asset_turnover_change,
        "steps": [
            ("Old Asset Turnover Ratio", asset_turnover_before),
            ("Adjusted Assets", adjusted_assets),
            ("New Asset Turnover Ratio", asset_turnover_after),
            ("Change in Asset Turnover Ratio", asset_turnover_change)
        ]
    }

    return structured_output


def template_debt_repayment_structured():
    """Structured version of Debt-to-Equity with Debt Changes template"""
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
    
    # Step 4: Calculate the change in Debt-to-Equity ratio
    debt_to_equity_change = round(debt_to_equity_after - debt_to_equity_before, 2)

    solution = (
        f"Step 1: Calculate the current Debt-to-Equity ratio:\n"
        f"  Debt-to-Equity Ratio = ${total_liabilities} / ${shareholders_equity} = {debt_to_equity_before:.2f}\n\n"
        f"Step 2: Calculate new liabilities after debt repayment and new borrowing:\n"
        f"  New Liabilities = ${total_liabilities} - ${debt_repayment} + ${new_borrowing} = ${new_liabilities}\n\n"
        f"Step 3: Calculate the new Debt-to-Equity ratio:\n"
        f"  New Debt-to-Equity Ratio = ${new_liabilities} / ${shareholders_equity} = {debt_to_equity_after:.2f}\n\n"
        f"Step 4: Calculate the change in Debt-to-Equity ratio:\n"
        f"  Change in Debt-to-Equity Ratio = {debt_to_equity_after:.2f} - {debt_to_equity_before:.2f} = {debt_to_equity_change:.2f}"
    )

    # Create the structured output
    structured_output = {
        "question": question,
        "original_answer": solution,
        "final_result": debt_to_equity_change,
        "steps": [
            ("Old Debt-to-Equity Ratio", debt_to_equity_before),
            ("New Liabilities", new_liabilities),
            ("New Debt-to-Equity Ratio", debt_to_equity_after),
            ("Change in Debt-to-Equity Ratio", debt_to_equity_change)
        ]
    }

    return structured_output


def template_quick_ratio_scenario_structured():
    """Structured version of Quick Ratio Scenario Analysis template"""
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
    existing_quick_assets = round(current_assets - inventories - prepaid_expenses, 2)
    additional_assets_needed = round(
        required_quick_assets - existing_quick_assets, 2
    )
    
    # Make sure additional_assets_needed is not negative
    additional_assets_needed = max(0, additional_assets_needed)

    solution = (
        f"Step 1: Calculate the existing quick ratio:\n"
        f"  Quick Ratio = (Current Assets - Inventories - Prepaid Expenses) / Current Liabilities\n"
        f"             = ({current_assets} - {inventories} - {prepaid_expenses}) / {current_liabilities} = {quick_ratio:.2f}\n\n"
        f"Step 2: Calculate the required quick assets to meet the minimum quick ratio:\n"
        f"  Required Quick Assets = Minimum Quick Ratio × Current Liabilities\n"
        f"                       = {min_quick_ratio:.2f} × {current_liabilities} = {required_quick_assets:.2f}\n\n"
        f"Step 3: Calculate the existing quick assets:\n"
        f"  Existing Quick Assets = Current Assets - Inventories - Prepaid Expenses\n"
        f"                       = {current_assets} - {inventories} - {prepaid_expenses} = {existing_quick_assets:.2f}\n\n"
        f"Step 4: Calculate the additional quick assets needed:\n"
        f"  Additional Quick Assets Needed = Required Quick Assets - Existing Quick Assets\n"
        f"                               = {required_quick_assets:.2f} - {existing_quick_assets:.2f} = {additional_assets_needed:.2f}"
    )

    # Create the structured output
    structured_output = {
        "question": question,
        "original_answer": solution,
        "final_result": additional_assets_needed,
        "steps": [
            ("Current Quick Ratio", quick_ratio),
            ("Required Quick Assets", required_quick_assets),
            ("Existing Quick Assets", existing_quick_assets),
            ("Additional Quick Assets Needed", additional_assets_needed)
        ]
    }

    return structured_output


def main():
    """Generate one sample from each template and write the results to a JSON file."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate one sample from each template
    asset_turnover_sample = template_asset_turnover_investment_depreciation_structured()
    debt_equity_sample = template_debt_repayment_structured()
    quick_ratio_sample = template_quick_ratio_scenario_structured()
    
    # Create a dictionary with all samples
    all_samples = {
        "asset_turnover_investment_depreciation": asset_turnover_sample,
        "debt_repayment": debt_equity_sample,
        "quick_ratio_scenario": quick_ratio_sample
    }
    
    # Use absolute path for output
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    output_file = os.path.join(project_root, "testset/financial_ratios/structured_samples.json")
    
    # Write all samples to a JSON file
    with open(output_file, "w") as file:
        json.dump(all_samples, file, indent=2)

    print(f"Successfully generated structured samples and saved to {output_file}")


if __name__ == "__main__":
    main() 