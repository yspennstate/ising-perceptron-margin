"""End-to-end meta-verification battery on the final committed tree:
sympy identities, then per certified margin the assembler, the
portable checker, every Lean kernel file, and the mutation battery.
One ledger, every exit code captured unmasked."""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
LEAN = os.path.expanduser(
    '~/.elan/toolchains/leanprover--lean4---v4.31.0/bin/lean.exe')
results = []


def run(name, args, cwd=REPO):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    ok = r.returncode == 0
    results.append((name, ok))
    tail = (r.stdout or r.stderr).strip().splitlines()
    print('%-42s %s   %s' % (name, 'PASS' if ok else 'FAIL',
                             tail[-1][:60] if tail else ''), flush=True)
    return ok


run('sympy identity battery', [sys.executable, 'sympy_identities.py'])
tags = sorted(d for d in os.listdir(
    os.path.join(REPO, 'lean_skeleton', 'generated'))
    if os.path.isdir(os.path.join(REPO, 'lean_skeleton',
                                  'generated', d)))
for tag in tags:
    run('assembler %s' % tag, [sys.executable, 'assemble_2varfn.py', tag])
    run('portable %s' % tag,
        [sys.executable, 'portable_check.py', '--tag', tag])
    gen = os.path.join(REPO, 'lean_skeleton', 'generated', tag)
    for f in sorted(os.listdir(gen)):
        if f.endswith('.lean'):
            run('lean %s/%s' % (tag, f), [LEAN, os.path.join(gen, f)])
    run('mutants %s' % tag,
        [sys.executable, 'mutate_margin.py', '--lean', LEAN, '--tag', tag],
        cwd=os.path.join(REPO, 'lean_skeleton'))

n_ok = sum(1 for _, ok in results if ok)
print('\nBATTERY: %d/%d PASS%s' % (n_ok, len(results),
      '' if n_ok == len(results) else '  <-- FAILURES ABOVE'), flush=True)
sys.exit(0 if n_ok == len(results) else 1)
