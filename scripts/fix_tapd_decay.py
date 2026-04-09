# Fix: wire decay into TAPD weights in run_mdt_seg.py and train_step
path = '/root/autodl-tmp/mkd-main/new-train/run_mdt_seg.py'
txt = open(path).read()

# Step1: rename base vars to TAPD names, keep old adv for compatibility
old_base = """    base_orth = getattr(config, 'alpha_orth', 0.1)
    base_adv  = getattr(config, 'alpha_adv_recon', 0.05)"""
new_base = """    base_orth  = getattr(config, 'alpha_orth', 0.1)
    base_adv   = getattr(config, 'alpha_adv_recon', 0.05)
    base_uni   = getattr(config, 'alpha_tapd_uni',   0.05)
    base_align = getattr(config, 'alpha_tapd_align', 0.1)"""

if old_base in txt:
    txt = txt.replace(old_base, new_base)
    print('[1] base vars updated')
else:
    print('[1] base vars NOT FOUND')

# Step2: extend decay block to include uni/align
old_decay = """        if epoch > 10:
            decay = 0.95 ** (epoch - 10)
            cur_orth = base_orth * decay
            cur_adv  = base_adv  * decay
        else:
            cur_orth = base_orth
            cur_adv  = base_adv"""
new_decay = """        if epoch > 10:
            decay     = 0.95 ** (epoch - 10)
            cur_orth  = base_orth  * decay
            cur_adv   = base_adv   * decay
            cur_uni   = base_uni   * decay
            cur_align = base_align * decay
        else:
            cur_orth  = base_orth
            cur_adv   = base_adv
            cur_uni   = base_uni
            cur_align = base_align"""

if old_decay in txt:
    txt = txt.replace(old_decay, new_decay)
    print('[2] decay block extended')
else:
    print('[2] decay block NOT FOUND')

# Step3: pass cur_uni/cur_align into train_step
old_call = "loss, _, _, ld = task.train_step(batch, alpha_orth=cur_orth, alpha_adv=cur_adv)"
new_call = "loss, _, _, ld = task.train_step(batch, alpha_orth=cur_orth, alpha_adv=cur_adv, alpha_uni=cur_uni, alpha_align=cur_align)"

if old_call in txt:
    txt = txt.replace(old_call, new_call)
    print('[3] train_step call updated')
else:
    print('[3] train_step call NOT FOUND')

open(path, 'w').write(txt)
print('run_mdt_seg.py done')

# Step4: update train_step signature and TAPD weight logic in tasks/mdt_seg.py
task_path = '/root/autodl-tmp/mkd-main/new-train/tasks/mdt_seg.py'
tcode = open(task_path).read()

old_sig = "    def train_step(self, batch, alpha_orth=None, alpha_adv=None):"
new_sig = "    def train_step(self, batch, alpha_orth=None, alpha_adv=None, alpha_uni=None, alpha_align=None):"

if old_sig in tcode:
    tcode = tcode.replace(old_sig, new_sig)
    print('[4] train_step signature updated')
else:
    print('[4] signature NOT FOUND')

old_weights = """        w_ortho = getattr(self.config, "alpha_tapd_ortho", 0.1)
        w_uni   = getattr(self.config, "alpha_tapd_uni",   0.05)
        w_align = getattr(self.config, "alpha_tapd_align", 0.1)"""
new_weights = """        w_ortho = alpha_orth  if alpha_orth  is not None else getattr(self.config, "alpha_tapd_ortho", 0.1)
        w_uni   = alpha_uni   if alpha_uni   is not None else getattr(self.config, "alpha_tapd_uni",   0.05)
        w_align = alpha_align if alpha_align is not None else getattr(self.config, "alpha_tapd_align", 0.1)"""

if old_weights in tcode:
    tcode = tcode.replace(old_weights, new_weights)
    print('[5] TAPD weights wired to dynamic decay')
else:
    print('[5] TAPD weights NOT FOUND')

open(task_path, 'w').write(tcode)
print('tasks/mdt_seg.py done')
