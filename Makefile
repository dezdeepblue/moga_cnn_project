
.PHONY: setup test smoke quick full full-cuda budget5 budget15 clean plots

setup:
	pip install -r requirements.txt

test:
	pytest -q

smoke:
	python -m src.pipeline --results-dir results_smoke --pop-size 4 --ngen 1 --eval-epochs 1 --num-seeds 1 --baseline-epochs 1 --baseline-max-combos 3 --retrain-epochs 1 --retrain-seeds 1 --retrain-k-each 1

quick:
	python -m src.pipeline --results-dir results_quick --pop-size 8 --ngen 2 --eval-epochs 1 --num-seeds 1 --retrain-epochs 3 --retrain-seeds 1 --copy-to-report

full:
	python -m src.pipeline --results-dir results --pop-size 24 --ngen 8 --eval-epochs 6 --num-seeds 3 --transfer-dataset cifar100 --retrain-epochs 50 --retrain-seeds 3 --copy-to-report

full-cuda:
	python -m src.pipeline --device cuda --results-dir results_cuda --pop-size 32 --ngen 10 --eval-epochs 8 --num-seeds 5 --transfer-dataset cifar100 --retrain-epochs 80 --retrain-seeds 3 --copy-to-report

budget5:
	python -m src.pipeline --results-dir results_budget5 --pop-size 24 --ngen 8 --eval-epochs 5 --num-seeds 3 --transfer-dataset cifar100 --retrain-epochs 50 --retrain-seeds 3 --copy-to-report

budget15:
	python -m src.pipeline --results-dir results_budget15 --pop-size 24 --ngen 8 --eval-epochs 15 --num-seeds 3 --transfer-dataset cifar100 --retrain-epochs 50 --retrain-seeds 3 --copy-to-report

plots:
	python -m src.plots --results-dir results --transfer-dataset cifar100

clean:
	rm -rf results results_* report/figures


.PHONY: report-tex report-pdf

report-tex:
	python -m src.report_gen --report-dir report --results-dir results --transfer-dataset cifar100
	python -m src.report_fill --results-dir results --transfer-dataset cifar100 --out report/REPORT_FINAL.md

report-pdf: report-tex
	cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
