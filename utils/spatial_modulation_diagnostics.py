"""Unified slot diagnostics. GT occupancy is recorded, never used in routing."""
import csv
import os
import torch

class ModulationCSVLogger:
    def __init__(self,directory,start_epoch=1):
        self.path=os.path.join(directory,'modulation_slots.csv')
        self.fields=['epoch','batch','event','scale','slot','accepted','updates','count','foreground_fraction','key_norm','gamma_rms','beta_rms','read_enabled','correction_rms_ratio','attention_mean','modulation_loss','foreground_fit_error','background_fit_error']
        retained=[]
        if os.path.exists(self.path):
            with open(self.path) as f:retained=[r for r in csv.DictReader(f) if int(r['epoch'])<start_epoch]
        with open(self.path,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=self.fields);writer.writeheader();writer.writerows(retained)
    @torch.no_grad()
    def snapshot(self,task,epoch,batch,event):
        rows=[]
        for i,s in enumerate(task.model.matrix.scales,1):
            b=s.bank;read=(task.matrix_read_stats or [{},{},{},{}])[i-1]
            for j in range(b.clusters):
                a=read.get('attention_mean',[])
                rows.append(dict(epoch=epoch,batch=batch,event=event,scale=i,slot=j,accepted=int(b.accepted),updates=int(b.updates),count=float(b.count[j]),foreground_fraction=float(b.fg_mass[j]/b.count[j]) if b.count[j]>0 else None,key_norm=float(b.keys[j].norm()),gamma_rms=float(b.gamma[j].square().mean().sqrt()),beta_rms=float(b.beta[j].square().mean().sqrt()),read_enabled=int(s.epoch)>=2 and bool(b.initialized),correction_rms_ratio=read.get('correction_rms_ratio'),attention_mean=a[j] if a else None,modulation_loss=task.mod_loss_sum/max(1,task.mod_loss_batches),foreground_fit_error=getattr(s,'last_supervision',{}).get('foreground_error'),background_fit_error=getattr(s,'last_supervision',{}).get('background_error')))
        with open(self.path,'a',newline='') as f:csv.DictWriter(f,fieldnames=self.fields).writerows(rows)
