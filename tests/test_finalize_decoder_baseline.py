import pytest
from scripts.finalize_decoder_baseline import select, finalize


def test_predeclared_selection_no_branch_best_splicing():
    assert select({'joint_dice':.77},{'joint_dice':.78})=='BN'
    assert select({'joint_dice':.78},{'joint_dice':.77})=='GN'
    assert select({'joint_dice':.77},{'joint_dice':.77})=='GN'


@pytest.mark.parametrize('state',['running','paused'])
def test_unfinished_training_must_not_freeze(tmp_path,state):
    (tmp_path/'status.json').write_text('{"state":"'+state+'"}')
    with pytest.raises(ValueError,match='finish successfully'):
        finalize(tmp_path,tmp_path,tmp_path,tmp_path)
    assert not (tmp_path/'frozen_training_config.json').exists()
