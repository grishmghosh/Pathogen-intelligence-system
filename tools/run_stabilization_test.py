from pathlib import Path
import json
import sys
from pathlib import Path as _P
# ensure workspace root on path
sys.path.insert(0, str(_P('.').resolve()))

from stabilization.schema_validator import validate_artifact_schemas, export_schema_validation_json, export_schema_validation_csv, export_schema_validation_txt

root = Path('results') / 'stabilization'
root.mkdir(parents=True, exist_ok=True)

experiment_summary_path = Path('results') / 'plots' / 'summaries' / 'experiment_summary.json'
report_manifest_path = Path('results') / 'reports' / 'report_step10_smoke' / 'manifests' / 'report_manifest.json'
report_summary_path = Path('results') / 'reports' / 'report_step10_smoke' / 'summaries' / 'report_summary.json'
trust_summary_path = Path('results') / 'disagreement' / 'trust_analysis' / 'trust_summary.json'

# load payloads if they exist, else use empty dicts
payloads = {}
for key, path in [('experiment_summary', experiment_summary_path), ('report_manifest', report_manifest_path), ('report_summary', report_summary_path), ('trust_summary', trust_summary_path)]:
    if path.exists():
        with path.open('r', encoding='utf-8') as h:
            try:
                payloads[key] = json.load(h)
            except Exception:
                payloads[key] = None
    else:
        payloads[key] = None

schema_result = validate_artifact_schemas(payloads)
print('compatibility summary before export:', schema_result.get('compatibility', {}).get('summary'))

schema_out = export_schema_validation_json(schema_result, root / 'schema_reports' / 'schema_validation.json')
print('written json:', schema_out)
export_schema_validation_csv(schema_result, root / 'schema_reports' / 'schema_validation.csv')
export_schema_validation_txt(schema_result, root / 'schema_reports' / 'schema_validation.txt')
print('exports complete')
