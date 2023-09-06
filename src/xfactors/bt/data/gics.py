from __future__ import annotations
import xtuples as xt

# ---------------------------------------------------------------

SECTOR_MAP_SHORT = {
    "10": "ENERGY",
    "15": "MATLS",
    "20": "INDLS",
    "25": "C-DISC",
    "30": "C-STAP",
    "35": "HEALTH",
    "40": "FINLS",
    "45": "INF-TECH",
    "50": "COM-SERV",
    "55": "UTILS",
    "60": "R-ESTATE",
}
SECTOR_MAP_SHORT = {
    "GICS {}".format(k): v for k, v in SECTOR_MAP_SHORT.items()
}

# ---------------------------------------------------------------

FULL_MAP = {
    "10": "ENERGY",
    # energy
    # "1010",
    # energy equipment
    "101010": "",
    # oil gas consumables
    "101020": "",
    "15": "MATERIALS",
    # materials
    # "1510",
    # chemicals
    "151010": "",
    # construction
    "151020": "",
    # containers
    "151030": "",
    # mining
    "151040": "",
    # paper
    "151050": "",
    "20": "INDUSTRIALS",
    # capital goods
    "2010": "",
    # comm and proff serices
    "2020": "",
    # transportion
    "2030": "",
    "25": "CONS DISC",
    # autos and comonents
    "2510": "",
    # consumer durable and apparalel
    "2520": "",
    # consumer services
    "2530": "",
    # cons disc distribution and retail
    "2550": "",
    "30": "CONS STAP",
    # cons stap dist and retail
    "3010": "",
    # food bev and tobacco
    "3020": "",
    # household and personal products
    "3030": "",
    "35": "HEALTH",
    # heathl care equip and services
    "3510": "",
    # pharam bio and life science
    "3520": "",
    "40": "FINANCIALS",
    # banks
    "4010": "",
    # financial services
    "4020": "",
    # insurance
    "4030": "",
    "45": "INFO TECH",
    # software
    "4510": "",
    # hardware
    "4520": "",
    # semis
    "4530": "",
    "50": "COMM SERV",
    # telco
    "5010": "",
    # media
    "5020": "",
    "55": "UTILITIES",
    "5510": "",
    "60": "REAL ESTATE",
    #reits
    "6010": "",
    # management and development
    # "6020",
}
FULL_MAP = {"GICS {}".format(k): v for k, v in FULL_MAP.items()}
FULL = xt.iTuple.from_keys(FULL_MAP)

SECTOR_MAP = {
    k: v for k, v in FULL_MAP.items()
    if len(k) == 2
}
SECTORS = xt.iTuple.from_keys(SECTOR_MAP)

DETAILED_MAP = {
    k: v for k, v in FULL_MAP.items()
    if len(k) > 2
}
DETAILED = xt.iTuple.from_keys(DETAILED_MAP)

# ---------------------------------------------------------------
