import math
import pytest
from scripts.diagnose_shared_ct_gradients import geometry, group_name


def test_alignment_and_adverse_direction_are_distinct():
    aligned=geometry(4.,4.,4.)
    assert aligned['cosine']==1 and aligned['cancellation_ratio']==1
    # Negative cosine alone does not imply either task is locally harmed.
    mild=geometry(-1.,4.,4.)
    assert mild['cosine']==-.25
    assert mild['missing_descent_dot']>0 and mild['full_descent_dot']>0
    # Stronger Full gradient cancels and reverses the Missing task direction.
    dominated=geometry(-2.,1.,4.)
    assert dominated['missing_descent_dot']<0 and dominated['full_descent_dot']>0
    assert math.isclose(dominated['full_to_missing_norm'],2)


def test_zero_and_nonfinite_geometry():
    assert geometry(0.,0.,0.)['cosine'] is None
    with pytest.raises(ValueError,match='Nonfinite'):
        geometry(float('nan'),1.,1.)


def test_actual_backbone_wrapper_stage_names():
    assert group_name('enc_ct.model.stages_2.blocks.0.conv_dw.weight')=='enc_ct.stages_2'
    assert group_name('enc_ct.model.stem_0.weight')=='enc_ct.stem_other'
    assert group_name('ct_align.projs.0.weight')=='ct_align'
