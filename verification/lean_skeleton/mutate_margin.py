"""Mutation battery for the margin Lean skeleton: corrupt one integer in
each generated file's data section and require the kernel check to FAIL.
A checker that accepts a corrupted certificate proves nothing.

Usage: python mutate_margin.py --lean <path-to-lean> [--out mutants]
"""
import argparse
import os
import re
import subprocess


def mutate_first_int(text):
    """Negate the largest integer in the data section: a sign flip
    falsifies every claim family here (packet positivity, corner-product
    determinants, chain seam equalities and nonemptiness), unlike a
    trailing-digit flip, which can land inside an inequality's margin
    and leave the mutated statement true."""
    body = text.split("theorem", 1)[0]
    body = re.sub(r'^set_option[^\n]*$', lambda m: ' ' * len(m.group(0)),
                  body, flags=re.M)      # options are not certificate data
    best = None
    for m in re.finditer(r'(?<=[(,\s])(-?\d+)(?=[),\s])', body):
        v = m.group(1)
        mag = int(v.lstrip('-'))
        if best is None or mag > best[0]:
            best = (mag, m.span(1), v)
    if best is None:
        raise ValueError('no mutable integer found')
    (_, (start, end), val) = best
    new_val = val[1:] if val.startswith('-') else '-' + val
    return text[:start] + new_val + text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lean', required=True)
    ap.add_argument('--src', default='generated')
    ap.add_argument('--out', default='mutants')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_ok = 0
    n_bad = 0
    for name in sorted(os.listdir(args.src)):
        if not name.endswith('.lean'):
            continue
        text = open(os.path.join(args.src, name), encoding='utf-8').read()
        mut = mutate_first_int(text)
        if mut == text:
            raise ValueError(f'{name}: mutation produced identical text')
        mpath = os.path.join(args.out, name)
        with open(mpath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(mut)
        r = subprocess.run([args.lean, mpath], capture_output=True,
                           text=True)
        rejected = r.returncode != 0
        print(f'{name}: mutant {"REJECTED" if rejected else "ACCEPTED"}')
        if rejected:
            n_ok += 1
        else:
            n_bad += 1
    print(f'{n_ok} mutants rejected, {n_bad} accepted')
    if n_bad or n_ok == 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
