import pytest
from scripts.evaluate_missing_rates_from_counts import simulate


def test_case_routing_endpoints_counts_and_nested_masks():
    rows=[]
    for case,slices in [('a',2),('b',1),('c',1),('d',1)]:
        for i in range(slices):
            rows.append({'image_id':f'{case}_{i}','case_id':case,
                         'full':dict(tp=2,fp=0,fn=0,hd95=1),
                         'missing':dict(tp=1,fp=1,fn=1,hd95=3)})
    r=simulate(rows,repeats=2)
    assert r['summary'][0]['dice_mean']==1
    assert r['summary'][-1]['dice_mean']==.5
    assert r['summary'][-1]['hd95_mean']==3
    for t in range(2):
        sets=[set(x['missing_cases']) for x in r['runs'] if x['trial']==t]
        assert all(a<=b for a,b in zip(sets,sets[1:]))
    assert r==simulate(rows,repeats=2)
    with pytest.raises(ValueError):simulate(rows,rates=[1.1])
