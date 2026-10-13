"""Two-epoch synthetic entry-point smoke, resume and selected-checkpoint eval."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_pet_correction_memory import FakeEncoder
from tasks.mdt_seg import MDTSegTeacher
from run_joint_pet_correction import main
from models.pet_correction_model import PETCorrectionModel


class TinyLoader(list):
    drop_last=True
    generator=torch.Generator()


class RunnerSmoke(unittest.TestCase):
    def test_train_save_resume_and_eval(self):
        torch.set_num_threads(1)
        def batch(prefix):
            gt=torch.zeros(4,1,32,32); gt[:,:,8:20,10:22]=1
            return {'ct':torch.randn(4,3,32,32),'pet':torch.randn(4,3,32,32),
                    'mask':gt,'image_id':[f'{prefix}_{i}' for i in range(4)],
                    'case_id':[f'{prefix}{i}' for i in range(4)]}
        loaders=(TinyLoader([batch('t'),batch('u')]),TinyLoader([batch('v')]),TinyLoader([batch('e')]))
        with tempfile.TemporaryDirectory() as tmp:
            args=['run_joint_pet_correction.py','--checkpoint_root',tmp,'--hash','contract',
                  '--pretrained','False','--epochs','2','--batch_size','4','--mixed_precision','False',
                  '--hd95_mode','surface','--decoder_channels','8','8','8','8','--num_workers','0',
                  '--ema_start_epoch','3','--bank_ema_start_epoch','0',
                  '--memory_capacity','16','--memory_key_dim','4','--memory_min_entries','1',
                  '--memory_topk','2','--memory_query_chunk','16']
            source_calls=[]
            original_candidates=PETCorrectionModel.make_candidates
            def trace_candidates(source,*a,**kw):
                source_calls.append((int(source.memory_step),bool(source.retrieval_active)))
                return original_candidates(source,*a,**kw)
            with patch('sys.argv',args),patch('run_joint_pet_correction._loaders',return_value=loaders),\
                 patch.object(PETCorrectionModel,'make_candidates',new=trace_candidates),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                main()
            self.assertEqual(source_calls,[(0,False),(1,False),(2,True),(3,True)])
            root=os.path.join(tmp,'MDT','contract')
            last=os.path.join(root,'ckpt.last.pth.tar')
            best=os.path.join(root,'ckpt.best_joint.pth.tar')
            ckpt=MDTSegTeacher.load_state_dicts(last)
            self.assertEqual(ckpt['epoch'],2)
            self.assertEqual(ckpt['global_batch_step'],4)
            self.assertEqual(int(ckpt['model']['memory_step']),4)
            self.assertTrue(ckpt['model']['retrieval_active'])
            self.assertEqual(ckpt['config']['norm_mode'],'imagenet')
            self.assertEqual(ckpt['config']['memory_warmup_epochs'],0)
            self.assertEqual(ckpt['config']['retrieval_start_epoch'],2)
            self.assertEqual(ckpt['eval_weights'],'raw')
            self.assertEqual(ckpt['ema_updates'],0)
            self.assertEqual(ckpt['bank_ema_updates'],4)
            self.assertTrue(ckpt['bank_ema_active'])
            self.assertEqual(ckpt['config']['ema_start_epoch'],3)
            self.assertEqual(ckpt['config']['bank_ema_start_epoch'],0)
            self.assertFalse(ckpt['config']['ct_aux_enabled'])
            self.assertFalse(any(k.startswith('ct_aux_decoder.') for k in ckpt['model']))
            with open(os.path.join(root,'memory_diagnostics.jsonl')) as f:
                records=[json.loads(line) for line in f]
            self.assertTrue(records)
            self.assertIn('source_rel_mean',records[-1]['queue']['0'])
            with open(os.path.join(root,'final_test_metrics.json')) as f:
                result=json.load(f)
            self.assertIn('memory_entries',result)
            self.assertEqual(result['hd95_mode'],'surface')
            self.assertEqual(result['missing_unit'],'slice')
            # Completed-run resume should restore and evaluate, not overwrite
            # queues or retrain another epoch.
            with patch('sys.argv',args+['--resume',last]),\
                 patch('run_joint_pet_correction._loaders',return_value=loaders),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                main()
            resumed=MDTSegTeacher.load_state_dicts(last)
            self.assertEqual(resumed['bank_ema_updates'],ckpt['bank_ema_updates'])
            for k,v in ckpt['model_bank_ema'].items():
                self.assertTrue(torch.equal(v,resumed['model_bank_ema'][k]),k)
            with patch('sys.argv',['run_joint_pet_correction.py','--eval_checkpoint',best,
                                  '--checkpoint_root',tmp,'--hash','eval']),\
                 patch('run_joint_pet_correction._loaders',return_value=loaders),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                main()
            self.assertTrue(os.path.isfile(os.path.join(tmp,'MDT','eval','reeval_test.json')))
            with patch('sys.argv',args+['--hash','skip','--epochs','1','--memory_diagnostics','False']),\
                 patch('run_joint_pet_correction._loaders',return_value=loaders),\
                 patch('run_joint_pet_correction.optimizer_step_succeeded',return_value=False),\
                 patch('models.dual_shared_add_baseline.create_feature_backbone',lambda *a,**k:FakeEncoder()):
                main()
            skipped=MDTSegTeacher.load_state_dicts(os.path.join(tmp,'MDT','skip','ckpt.last.pth.tar'))
            self.assertEqual(int(skipped['model']['memory_step']),0)
            self.assertEqual(skipped['ema_updates'],0)
            self.assertEqual(skipped['bank_ema_updates'],0)
            for k,v in skipped['model'].items():
                if k.endswith('.memory.valid'):
                    self.assertFalse(v.any())


if __name__=='__main__':
    unittest.main()
