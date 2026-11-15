import sys, re
for fn in sys.argv[1:]:
    t = open(fn).read().splitlines(); cur = None; r = {}
    for l in t:
        m = re.match(r"\|(\w+)\s*\|", l)
        if m and m.group(1) in ("arc_challenge","arc_easy","boolq","hellaswag","piqa","winogrande"): cur = m.group(1)
        c = [x.strip() for x in l.split("|")]
        if cur and len(c) > 7 and c[5] in ("acc","acc_norm"): r.setdefault(cur, {})[c[5]] = float(c[7])
    use = {"arc_challenge":"acc_norm","arc_easy":"acc_norm","hellaswag":"acc_norm","piqa":"acc_norm","boolq":"acc","winogrande":"acc"}
    v = [r[k][use[k]] for k in ["arc_challenge","arc_easy","boolq","hellaswag","piqa","winogrande"] if k in r]
    print(fn.split("/")[-1], [round(x,3) for x in v], "avg %.4f" % (sum(v)/len(v)) if len(v) == 6 else "incomplete")
