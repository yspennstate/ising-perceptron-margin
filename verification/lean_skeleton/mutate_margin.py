"""Mutation battery for the margin Lean skeleton: corrupt one integer in
each generated file's data section and require the kernel check to FAIL.
A checker that accepts a corrupted certificate proves nothing.

Usage: python mutate_margin.py --lean <path-to-lean> [--out mutants]
"""
import argparse
import os
import re
import subprocess


def mutations(text, k=4):
    """Yield up to k mutants, each negating one of the k largest
    integers in the data section.  A single sign flip can land on the
    favorable side of a one-sided inequality (negating the REQUIRED
    radius leaves inradius > required true - seen at tag 0p13), so the
    battery tries the top-k in turn and passes when ANY mutant is
    rejected; only all-survive is an alarm."""
    body = text.split("theorem", 1)[0]
    body = re.sub(r'^set_option[^\n]*$', lambda m: ' ' * len(m.group(0)),
                  body, flags=re.M)      # options are not certificate data
    found = []
    for m in re.finditer(r'(?<=[(,\s])(-?\d+)(?=[),\s])', body):
        v = m.group(1)
        found.append((int(v.lstrip('-')), m.span(1), v))
    if not found:
        raise ValueError('no mutable integer found')
    found.sort(key=lambda t: -t[0])
    for (_, (start, end), val) in found[:k]:
        new_val = val[1:] if val.startswith('-') else '-' + val
        yield text[:start] + new_val + text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lean', required=True)
    ap.add_argument('--tag', default='0p05')
    ap.add_argument('--src', default=None,
                    help='source dir (default: generated/<tag>)')
    ap.add_argument('--out', default=None,
                    help='mutant dir (default: mutants/<tag>)')
    args = ap.parse_args()
    if args.src is None:
        args.src = os.path.join('generated', args.tag)
    if args.out is None:
        args.out = os.path.join('mutants', args.tag)
    os.makedirs(args.out, exist_ok=True)
    n_ok = 0
    n_bad = 0
    for name in sorted(os.listdir(args.src)):
        if not name.endswith('.lean'):
            continue
        text = open(os.path.join(args.src, name), encoding='utf-8').read()
        rejected = False
        tried = 0
        for mut in mutations(text):
            tried += 1
            mpath = os.path.join(args.out, name)
            with open(mpath, 'w', encoding='utf-8', newline='\n') as f:
                f.write(mut)
            r = subprocess.run([args.lean, mpath], capture_output=True,
                               text=True)
            if r.returncode != 0:
                rejected = True
                break
        print(f'{name}: mutant {"REJECTED" if rejected else "ACCEPTED"}'
              f' (tried {tried})')
        if rejected:
            n_ok += 1
        else:
            n_bad += 1
    print(f'{n_ok} mutants rejected, {n_bad} accepted')
    if n_bad or n_ok == 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
