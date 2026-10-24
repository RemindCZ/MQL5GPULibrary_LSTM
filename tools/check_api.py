"""Check C/CUDA/MQL signatures without requiring CUDA or MetaEditor."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def signatures(text, pattern, mql=False):
    result = {}
    for return_type, name, args in re.findall(pattern, text, re.S):
        types = []
        for argument in args.split(','):
            argument = argument.strip()
            if not argument or argument == 'void':
                continue
            match = re.fullmatch(r'(const\s+)?(int|double|char|short)\s*([*&]?)\s*(\w+)\s*(\[\])?', argument)
            if not match:
                raise ValueError(f'{name}: cannot parse {argument!r}')
            const, dtype, pointer, param_name, array = match.groups()
            if mql and pointer:
                expected_array = param_name in ('X', 'T', 'Y', 'buf', 'parents')
                if bool(array) != expected_array:
                    raise ValueError(f'{name}: incorrect MQL array/scalar reference for {param_name}')
            types.append((bool(const), dtype, bool(pointer)))
        result[name] = (return_type.replace('MQL_BOOL', 'int'), types)
    return result


def main():
    source = signatures((ROOT / 'kernel.cu').read_text(encoding='utf-8'),
                        r'DLL_EXPORT\s+(\w+)\s+DLL_CALL\s+(DN_\w+)\(([^{}]*?)\)\s*try\s*\{')
    header = signatures((ROOT / 'include/MQL5GPULibrary.h').read_text(),
                        r'DN_API\s+(\w+)\s+DN_CALL\s+(DN_\w+)\((.*?)\);')
    mql = signatures((ROOT / 'MQL5/Include/MQL5GPULibrary.mqh').read_text(),
                     r'(\w+)\s+(DN_\w+)\((.*?)\);', mql=True)
    if not source or source != header or source != mql:
        raise SystemExit('CUDA exports, C header and MQL imports differ')
    for path in (ROOT / 'MQL5').rglob('*.mq5'):
        text = path.read_text(encoding='utf-8-sig')
        unknown = set(re.findall(r'\b(DN_\w+)\s*\(', text)) - source.keys()
        if unknown or '#include <MQL5GPULibrary.mqh>' not in text:
            raise SystemExit(f'{path}: missing shared include or unknown calls {unknown}')
    print(f'{len(source)} exports match C and MQL declarations; examples use the shared include.')


if __name__ == '__main__':
    main()
