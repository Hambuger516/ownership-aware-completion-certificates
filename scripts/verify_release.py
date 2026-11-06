"""Verify the public file manifest and private-information exclusions without printing values."""
from pathlib import Path
import hashlib,json,re,sys
R=Path(__file__).resolve().parents[1];j=json.loads((R/'provenance/RELEASE_MANIFEST.json').read_text());bad=[]
for name,h in j['files_sha256'].items():
 p=R/name
 if not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append({'file':name,'category':'missing_or_modified'})
patterns={'private_path':r'(?<![\w/])/(?:Users|home|root|data|workspace)/[^\s]+','private_deployment_host':r'connect\.[\w.-]+|\b(?:bm-[a-z0-9]+|g[0-9]{4,})\b','private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----','credential_token':r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}'}
for name in j['files_sha256']:
 p=R/name
 try:s=p.read_text()
 except UnicodeError:continue
 for category,pat in patterns.items():
  if re.search(pat,s):bad.append({'file':name,'category':category})
print(json.dumps({'files_checked':len(j['files_sha256']),'findings':bad,'values_printed':False},indent=2));raise SystemExit(bool(bad))
