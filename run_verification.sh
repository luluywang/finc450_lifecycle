#!/bin/bash
# Verification script for Python vs TypeScript implementation comparison
# Usage: ./run_verification.sh

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "Python vs TypeScript Verification Suite"
echo "================================================"
echo ""

# Step 1: Generate Python verification data
echo "Step 1: Generating Python verification data..."
python3 compare_implementations.py --pretty -o output/python_verification.json
echo "   ✓ Saved to output/python_verification.json"
echo ""

# Step 2: Check if TypeScript data exists
if [ -f "output/typescript_verification.json" ]; then
    echo "Step 2: Found existing TypeScript verification data"
    echo ""

    # Step 3: Run comparison
    echo "Step 3: Running comparison..."
    python3 compare_implementations.py --compare output/typescript_verification.json -o output/verification_report.md
    echo "   ✓ Generated output/verification_report.md"
    echo ""

    # Display summary
    echo "================================================"
    echo "VERIFICATION REPORT SUMMARY"
    echo "================================================"
    head -20 output/verification_report.md
    echo ""
    echo "Full report saved to: output/verification_report.md"
else
    echo "Step 2: TypeScript verification data not found"
    echo ""
    echo "To generate TypeScript data:"
    echo "  1. Start the dev server: cd web/app && npm run dev"
    echo "  2. Open http://localhost:5173/ in your browser"
    echo "  3. Click 'Export Verification Data' button in the sidebar"
    echo "  4. Save the downloaded file as output/typescript_verification.json"
    echo "  5. Re-run this script"
fi

echo ""
echo "================================================"
echo "Quick Verification Tests"
echo "================================================"

echo ""
echo "Python core function tests:"
python3 -c "
from core.economics import effective_duration, zero_coupon_price
from core import EconomicParams

econ = EconomicParams()
print(f'  effective_duration(20, 0.85) = {effective_duration(20, 0.85):.6f}')
print(f'  effective_duration(20, 1.0)  = {effective_duration(20, 1.0):.6f}')
print(f'  zero_coupon_price(0.02, 20, 0.02, 1.0) = {zero_coupon_price(0.02, 20, 0.02, 1.0):.6f}')
print(f'  mu_bond = {econ.mu_bond:.6f}')
"

echo ""
echo "MV Optimization test:"
python3 -c "
from core.economics import compute_mv_optimal_allocation
from core import EconomicParams, LifecycleParams

e = EconomicParams()
p = LifecycleParams()
result = compute_mv_optimal_allocation(e.mu_excess, e.mu_bond, e.sigma_s, e.sigma_r, e.rho, e.bond_duration, p.gamma)
print(f'  target_stock = {result[0]:.6f}')
print(f'  target_bond  = {result[1]:.6f}')
print(f'  target_cash  = {result[2]:.6f}')
"

echo ""
echo "Done!"
