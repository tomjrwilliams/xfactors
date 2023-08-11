set MYPYPATH=C:/hc/xtuples/src/xtuples;C:/hc/xfactors;
python -m mypy ./ --check-untyped-defs --soft-error-limit=-1 | python ./tests/filter_mypy.py