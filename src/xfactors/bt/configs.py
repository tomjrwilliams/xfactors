
INDICES_OLD = [
    "CAC Index",
    "DAX Index",
    "UKX Index",
    "SMI Index",
    "OMX Index",
    "IBEX Index",
    "IT30 Index",
    # "FTSEMIB Index",
    "PSI20 Index",
    "AEX Index",
    # "ISEQ Index",
    "BEL20 Index",
    "KFX Index",
    "OBX Index",
]

INDICES = core.Array(INDICES_OLD + [
    # "SXXP Index",
    # "SPX Index",
    "SX5E Index",
    "SX7E Index",
])

INDICES_MAJOR = core.Array([

    "SX5E Index",
    "SX7E Index",

    "CAC Index",
    "DAX Index",
    "UKX Index",
    
    "SMI Index",
    "OMX Index",

])



GICS_BREAKDOWN = core.Array([
    #energy
    "10",
    # energy
    # "1010",
    # energy equipment
    "101010",
    # oil gas consumables
    "101020",
    # materials
    "15",
    # materials
    # "1510",
    # chemicals
    "151010",
    # construction
    "151020",
    # containers
    "151030",
    # mining
    "151040",
    # paper
    "151050",
    # industrials
    "20",
    # capital goods
    "2010",
    # comm and proff serices
    "2020",
    # transportion
    "2030",
    # cons disc
    "25",
    # autos and comonents
    "2510",
    # consumer durable and apparalel
    "2520",
    # consumer services
    "2530",
    # cons disc distribution and retail
    "2550",
    # cons stap
    "30",
    # cons stap dist and retail
    "3010",
    # food bev and tobacco
    "3020",
    # household and personal products
    "3030",
    # health
    "35",
    # heathl care equip and services
    "3510",
    # pharam bio and life science
    "3520",
    # fins
    "40",
    # banks
    "4010",
    # financial services
    "4020",
    # insurance
    "4030",
    # it
    "45",
    # software
    "4510",
    # hardware
    "4520",
    # semis
    "4530",
    # comm services
    "50",
    # telco
    "5010",
    # media
    "5020",
    # utilities
    "55",
    "5510",
    # real estate
    "60",
    #reits
    "6010",
    # management and development
    # "6020",
])

GICS_SECTORS = GICS_BREAKDOWN.filter(lambda s: len(s) == 2).map(lambda s: "GICS {}".format(s))
GICS_INDUSTRY_GROUPS = GICS_BREAKDOWN.filter(lambda s: len(s) > 2).map(lambda s: "GICS {}".format(s))
