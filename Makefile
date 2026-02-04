# Makefile for FINC450 Lifecycle Investment Strategy
#
# Usage:
#   make          - Generate all figures and PDFs
#   make figures  - Generate lecture figures only
#   make pdfs     - Generate PDF reports only
#   make clean    - Remove all generated files

PYTHON = python3
OUTPUT_DIR = output
FIGURES_DIR = $(OUTPUT_DIR)/figures
FIGURE_FORMAT = png
FIGURE_DPI = 150

# Output files
LIFECYCLE_PDF = $(OUTPUT_DIR)/lifecycle_report.pdf
DASHBOARD_PDF = $(OUTPUT_DIR)/strategy_comparison.pdf
TEACHING_PDF = $(OUTPUT_DIR)/teaching_scenarios.pdf
SINGLE_DRAW_PDF = $(OUTPUT_DIR)/single_draw.pdf

# Source files that trigger rebuilds
CORE_SOURCES = core/params.py core/economics.py core/simulation.py core/strategies.py
VIZ_SOURCES = visualization/styles.py visualization/helpers.py \
              visualization/lifecycle_plots.py visualization/monte_carlo_plots.py \
              visualization/comparison_plots.py visualization/sensitivity_plots.py

.PHONY: all figures pdfs teaching single-draw clean help

# Default target: build everything
all: figures pdfs

# Generate all lecture figures
figures: $(FIGURES_DIR)/.stamp

$(FIGURES_DIR)/.stamp: generate_lecture_figures.py $(CORE_SOURCES) $(VIZ_SOURCES) | $(FIGURES_DIR)
	@echo "Generating lecture figures..."
	$(PYTHON) generate_lecture_figures.py --output-dir $(FIGURES_DIR) --format $(FIGURE_FORMAT) --dpi $(FIGURE_DPI)
	@touch $@

# Generate all PDFs
pdfs: $(LIFECYCLE_PDF) $(DASHBOARD_PDF) $(TEACHING_PDF)

# Teaching scenarios PDF
teaching: $(TEACHING_PDF)

# Lifecycle report PDF
$(LIFECYCLE_PDF): generate_report.py $(CORE_SOURCES) $(VIZ_SOURCES) | $(OUTPUT_DIR)
	@echo "Generating $(LIFECYCLE_PDF)..."
	$(PYTHON) generate_report.py -o $@

# Strategy comparison PDF
$(DASHBOARD_PDF): compare_strategies.py $(CORE_SOURCES) $(VIZ_SOURCES) | $(OUTPUT_DIR)
	@echo "Generating $(DASHBOARD_PDF)..."
	$(PYTHON) compare_strategies.py
	@mv strategy_comparison.pdf $@

# Teaching scenarios PDF
$(TEACHING_PDF): compare_teaching_scenarios.py $(CORE_SOURCES) $(VIZ_SOURCES) | $(OUTPUT_DIR)
	@echo "Generating $(TEACHING_PDF)..."
	$(PYTHON) compare_teaching_scenarios.py -o $@

# Single random draw analysis
single-draw: $(SINGLE_DRAW_PDF)

$(SINGLE_DRAW_PDF): generate_single_draw.py generate_rebalancing_demo.py $(CORE_SOURCES) $(VIZ_SOURCES) | $(OUTPUT_DIR)
	@echo "Generating $(SINGLE_DRAW_PDF)..."
	$(PYTHON) generate_single_draw.py -o $@

# Create output directories
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(FIGURES_DIR):
	mkdir -p $(FIGURES_DIR)

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)/figures
	rm -f $(OUTPUT_DIR)/*.pdf
	rm -f *.aux *.log *.nav *.out *.snm *.toc
	rm -rf __pycache__ */__pycache__
	@echo "Done."

# Help
help:
	@echo "FINC450 Lifecycle Investment Strategy - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Generate all figures and PDFs (default)"
	@echo "  figures  - Generate lecture figures ($(FIGURES_DIR)/)"
	@echo "  pdfs     - Generate PDF reports"
	@echo "  teaching    - Generate teaching scenarios PDF only"
	@echo "  single-draw - Generate single random draw analysis"
	@echo "  clean       - Remove all generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Output files:"
	@echo "  $(LIFECYCLE_PDF)"
	@echo "  $(DASHBOARD_PDF)"
	@echo "  $(TEACHING_PDF)"
	@echo "  $(SINGLE_DRAW_PDF)"
	@echo "  $(FIGURES_DIR)/*.png"
