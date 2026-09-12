import copy
import json
from pathlib import Path
import unittest
import numpy as np
from pooled_muscle_mapping import make_mapping
from neuromuscular import NamedMuscleDrive


class TestPooledMapping(unittest.TestCase):
    def inputs(self):
        root=Path(__file__).parent
        return json.loads((root/'muscle-probe-002.json').read_text()),json.loads((root/'motor-annotations.json').read_text())

    def test_coverage_sharing_and_zero(self):
        inventory,annotation=self.inputs();report=make_mapping(inventory,annotation)
        self.assertEqual(len(report['muscles']),15)
        rows=sorted(r['cns_row'] for r in annotation['motors'])
        drive=NamedMuscleDrive(report,rows)
        self.assertTrue(all(m['candidate_motor_rows'] for m in report['muscles']))
        np.testing.assert_array_equal(drive(np.zeros(len(rows))),0)
        np.testing.assert_allclose(drive(np.full(len(rows),.2)),.2,atol=3e-8)
        byname={m['muscle']:m for m in report['muscles']}
        self.assertEqual(byname['LFF_trochanter_flexor_a']['candidate_motor_rows'],byname['LFF_trochanter_flexor_b']['candidate_motor_rows'])
        rates=np.zeros(len(rows))
        rates[[rows.index(r) for r in byname['LFTibia_flex_93434']['candidate_motor_rows']]]=.5
        output=drive(rates);target=byname['LFTibia_flex_93434']['muscle_id']
        self.assertAlmostEqual(float(output[target]),.5)
        np.testing.assert_array_equal(np.delete(output,target),0)

    def test_identity_and_missing_pool_rejected(self):
        i,a=self.inputs();bad=copy.deepcopy(a);bad['source_sha256']='wrong'
        with self.assertRaises(ValueError):make_mapping(i,bad)
        bad=copy.deepcopy(a);bad['motors']=[r for r in bad['motors'] if r['mancType']!='Sternotrochanter MN']
        with self.assertRaises(ValueError):make_mapping(i,bad)


if __name__=='__main__':unittest.main()
