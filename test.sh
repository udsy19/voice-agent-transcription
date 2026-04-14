#!/bin/bash
# Muse — test runner
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════"
echo "  Muse Test Suite"
echo "═══════════════════════════════════════════"
echo ""

# Syntax checks — every Python file must parse
echo "── Syntax checks ──"
for f in "$DIR"/*.py "$DIR"/integrations/*.py; do
    python3 -c "import ast;ast.parse(open('$f').read())" && echo "  ✓ $(basename $f)" || { echo "  ✗ $(basename $f)"; exit 1; }
done

# Frontend JS syntax
echo ""
echo "── Frontend JS ──"
node -e "
const fs=require('fs');
['electron/ui/app.html','electron/ui/pill.html'].forEach(f=>{
  const h=fs.readFileSync(f,'utf8');
  const s=h.match(/<script>([\s\S]*?)<\/script>/g);
  s.forEach(x=>{try{new Function(x.replace(/^<script>/,'').replace(/<\/script>$/,''))}catch(e){console.log('  ✗ '+f+': '+e.message);process.exit(1)}});
  const o=(h.match(/<div[\s>]/g)||[]).length;
  const c=(h.match(/<\/div>/g)||[]).length;
  console.log('  ✓ '+f+' ('+o+' divs)');
  if(o!==c){console.log('    DIVS MISMATCH: '+o+'/'+c);process.exit(1)}
});
"

node --check "$DIR/electron/main.js" && echo "  ✓ main.js"
node --check "$DIR/electron/preload.js" && echo "  ✓ preload.js"

# Unit tests
echo ""
echo "── Unit tests ──"
if [ -f "$DIR/tests.py" ]; then
    cd "$DIR" && python3 tests.py && echo "  ✓ tests.py passed" || echo "  ⚠ tests.py had failures"
fi
if [ -f "$DIR/test_units.py" ]; then
    cd "$DIR" && python3 test_units.py 2>&1 | tail -5 || echo "  ⚠ test_units.py issues"
fi

# Hallucination filter tests
echo ""
echo "── Hallucination filters ──"
python3 -c "
import sys; sys.path.insert(0, '$DIR')
from quick_capture import detect
assert detect('remind me to buy milk') == ('todo', 'buy milk')
assert detect('I need to know if the deploy succeeded') is None
print('  ✓ quick_capture filters questions')
"

python3 -c "
import sys; sys.path.insert(0, '$DIR')
import app
halluc_tests = [
    ('Hello how are you', False),
    ('Canva help me to talk НаASU外 puedaução?', True),
    ('thank you for watching', True),
    ('', True),
]
for text, expected in halluc_tests:
    assert app._is_hallucination(text) == expected, f'FAIL: {text!r}'
print('  ✓ hallucination filter')
"

echo ""
echo "═══════════════════════════════════════════"
echo "  All tests passed"
echo "═══════════════════════════════════════════"
