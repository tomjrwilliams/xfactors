
import xtuples as xt

# ---------------------------------------------------------------

FULL_MAP = {
    # USD
    "YCSW0023 Index": "USD-S",
    "YCGT0025 Index": "USD-G",
    "YCGT0169 Index": "USD-I",
    # # EUR
    "YCSW0045 Index": "EUR-S",
    "YCSW0092 Index": "EUR-S",
    # USD CORP IG CUrve
    "BVSC0076 Index": "USD-IG",
    # AA
    "BVSC0073 Index": "USD-AA",
    # A
    "BVSC0074 Index": "USD-A",
    # bbb
    "BVSC0075 Index": "USD-BBB",
    # bb
    "BVSC0193 Index": "USD-BB",
    # b
    "BVSC0195 Index": "USD-B",
    # EUR
    # aa
    "BVSC0165 Index": "EUR-AA",
    # a
    "BVSC0077 Index": "EUR-A",
    # bbb
    "BVSC0166 Index": "EUR-BBB",
    # JP
    # aa
    "BVSC0153 Index": "JPY-AA",
    # a
    "BVSC0154 Index": "JPY-A",
    # # DE
    "YCGT0016 Index": "EUR-DE",
    # # FR
    "YCGT0014 Index": "EUR-FR",
    # # JP
    "YCGT0018 Index": "JPY-G",
    "YCSW0097 Index": "JPY-S",
    "YCGT0385 Index": "JPY-I",
    # # UK
    "YCGT0022 Index": "GBP-G",
    "YCSW0022 Index": "GBP-S",
    # # AU
    "YCGT0001 Index": "AUD-G",
    "YCSW0001 Index": "AUD-S",
    "YCGT0204 Index": "AUD-I",
    # # IT
    "YCGT0040 Index": "EUR-IT",
    # "YCGT0331 Index": "", inflation?
    # # CA
    "YCGT0007 Index": "CAD-G",
    # # CN
    "YCGT0299 Index": "CHN-G",
    # # SP
    "YCGT0061 Index": "EUR-ES",
    # # SW
    "YCGT0082 Index": "CHF-G",
    # # SE
    "YCGT0021 Index": "SEK-G",
    # NZ
    "YCGT0049 Index": "NZD-G",
}
FULL_MAP = {
    k.split(" ")[0]: v for k, v in FULL_MAP.items()
}
FULL = xt.iTuple.from_keys(FULL_MAP)

def is_corp(ccy, suffix, *, with_ccy = None):
    return any([
        rating in suffix for rating in ["A", "B"]
    ]) and (
        True if not with_ccy else ccy == with_ccy
    )

CORP_USD_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="USD")
}
CORP_USD = xt.iTuple.from_keys(CORP_USD_MAP)

CORP_EUR_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="EUR")
}
CORP_EUR = xt.iTuple.from_keys(CORP_EUR_MAP)

CORP_JPY_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="JPY")
}
CORP_JPY = xt.iTuple.from_keys(CORP_JPY_MAP)

# ---------------------------------------------------------------
