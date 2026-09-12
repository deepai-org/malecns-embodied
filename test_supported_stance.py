"""Check the executed search's recruitment compression, not force feasibility."""
import json
from pathlib import Path
import unittest
import numpy as np


class RecruitmentCompressionTest(unittest.TestCase):
    def test_recorded_grouping_preserves_named_input_set(self):
        root=Path(__file__).parent
        report=json.loads((root/'evidence/supported-stance-search-001.json').read_text())
        reference=json.loads((root/'evidence/trials/six-leg-stance-feedback-001/result.json').read_text())
        motor=json.loads((root/'motor-annotations.json').read_text())['motors']
        weights=np.zeros((90,len(motor)))
        for i,pool in enumerate(reference['recruitment']):
            ids=[j for j,n in enumerate(motor) if n['cns_row'] in pool['rows']]
            if ids:weights[i,ids]=1/len(ids)
        groups={}
        for j in range(len(motor)):
            if np.any(weights[:,j]):groups.setdefault(tuple(weights[:,j]),[]).append(j)
        grouped=np.stack([np.asarray(c)*len(ids) for c,ids in groups.items()],axis=1)
        np.testing.assert_array_equal(grouped,report['grouped_recruitment'])
        rng=np.random.default_rng(731)
        for _ in range(100):
            inputs=rng.uniform(size=len(motor))
            means=np.array([inputs[ids].mean() for ids in groups.values()])
            np.testing.assert_allclose(weights@inputs,grouped@means,rtol=0,atol=1e-14)
        for result in report['results']:
            controls=np.asarray(result['group_inputs'])
            self.assertTrue(np.all((0<=controls)&(controls<=1)))
            np.testing.assert_allclose(grouped@controls,result['activations'],rtol=0,atol=1e-14)


if __name__=='__main__':unittest.main()
