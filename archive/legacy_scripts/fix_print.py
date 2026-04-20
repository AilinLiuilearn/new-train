path = '/root/autodl-tmp/mkd-main/new-train/run_mdt_seg.py'
txt = open(path).read()

old = """            if (i + 1) % 50 == 0:
                orth_str = ' orth={:.4f}'.format(ld['loss_orth'].item()) if 'loss_orth' in ld else ''
                adv_str  = ' adv={:.4f}'.format(ld['loss_adv'].item())  if 'loss_adv'  in ld else ''
                print('  Ep{}[{}/{}] loss={:.4f} seg={:.4f}{}{}'.format(
                    epoch, i+1, spe, loss.item(), ld['loss_seg'].item(), orth_str, adv_str))"""

new = """            if (i + 1) % 50 == 0:
                tapd_str  = ' tapd={:.4f}'.format(ld['loss_tapd'].item())  if 'loss_tapd'  in ld else ''
                orth_str  = ' orth={:.4f}'.format(ld['loss_ortho'].item()) if 'loss_ortho' in ld else ''
                uni_str   = ' uni={:.4f}'.format(ld['loss_uni'].item())    if 'loss_uni'   in ld else ''
                align_str = ' aln={:.4f}'.format(ld['loss_align'].item())  if 'loss_align' in ld else ''
                print('  Ep{}[{}/{}] loss={:.4f} seg={:.4f}{}{}{}{}'.format(
                    epoch, i+1, spe, loss.item(), ld['loss_seg'].item(),
                    tapd_str, orth_str, uni_str, align_str))"""

if old in txt:
    open(path, 'w').write(txt.replace(old, new))
    print('OK: print updated')
else:
    print('NOT FOUND - checking current print block:')
    idx = txt.find('if (i + 1) % 50 == 0')
    print(repr(txt[idx:idx+300]))
