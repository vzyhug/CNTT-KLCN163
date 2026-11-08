from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'data_parts/manifest.json').read_text())
output=root/manifest['output']
temporary=output.with_suffix('.npz.partial')
h=hashlib.sha256()
with temporary.open('wb') as target:
    for item in manifest['parts']:
        block=(root/'data_parts'/item['name']).read_bytes()
        if hashlib.sha256(block).hexdigest()!=item['sha256']:
            raise RuntimeError('Missing or damaged LFS part: '+item['name']+'; run git lfs pull')
        target.write(block);h.update(block)
if h.hexdigest()!=manifest['sha256']: raise RuntimeError('Dataset checksum mismatch')
temporary.replace(output)
print('Restored and verified:',output)
