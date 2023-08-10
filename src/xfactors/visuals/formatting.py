
# ---------------------------------------------------------------

def unindent(v):
    ls = v.split("\n")
    shared_padding = min([
        len(l) - len(l.lstrip()) for l in ls if len(l.strip())
    ])
    return "\n".join([l[shared_padding:] for l in ls])

def left_pad(v, l=1, pad=" "):
    to_add = max([l - len(v), 0])
    if to_add == 0:
        return v
    if "-" in v:
        return "-" + (to_add * pad) + v[1:]
    return (to_add * pad) + v

def right_pad(v, l=1, pad=" "):
    to_add = max([l - len(v), 0])
    if to_add == 0:
        return v
    return v + (to_add * pad)

def pad(v, v_len, dp = None):
    if dp is None:
        v = str(v)
    elif dp == 0:
        v = str(round(v))
    else:
        v = str(round(v, dp))
    if "-" in v:
        return left_pad(v, v_len, pad="0")
    else:
        return left_pad(v, v_len - 1, pad="0")
        
# ---------------------------------------------------------------
