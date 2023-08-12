set MYPYPATH=./__local__;C:/xtuples/src/xtuples;C:/xfactors;
python -m mypy .%1 --check-untyped-defs --soft-error-limit=-1 | python ./tests/filter_mypy.py