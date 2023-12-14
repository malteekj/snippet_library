#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:49:04 2023

@author: maltejensen
"""
import matplotlib.pyplot as plt

fig, axx = plt.subplots(1,3, figsize=(15,5))

cmap = plt.get_cmap("tab10")
alpha = 0.6
markersize = 9

avg_mean = (flat(all_perc_ID_mean_ROI)+flat(all_perc_ID_mean_pred))/2
diff_mean = flat(all_perc_ID_mean_pred)-flat(all_perc_ID_mean_ROI)

avg_max = (flat(all_perc_ID_max_ROI)+flat(all_perc_ID_max_pred))/2
diff_max = flat(all_perc_ID_max_pred)-flat(all_perc_ID_max_ROI)

avg_vol = (flat(all_ROI_vol)+flat(all_pred_vol))/2
diff_vol = flat(all_pred_vol)-flat(all_ROI_vol)


# vol
width = np.abs(avg_vol.min()-avg_vol.max())
start, end = avg_vol.min()-width*0.1, avg_vol.max()+width*0.1
# plt.figure()
axx[0].plot(avg_vol, diff_vol, '.', color=cmap(0), markersize=markersize)
axx[0].plot([start, end], [diff_vol.mean()]*2,'--', color=cmap(1), label='mean diff', alpha=alpha)
axx[0].plot([start, end], [diff_vol.mean()+diff_vol.std()*1.96]*2,'--', color=cmap(2), 
         label='95% conf', alpha=alpha)
axx[0].plot([start, end], [diff_vol.mean()-diff_vol.std()*1.96]*2,'--', color=cmap(2), alpha=alpha)
axx[0].set_title('Bland–Altman plot tumor volume (mL)')
axx[0].set_xlabel('Average')
axx[0].set_ylabel('difference')
axx[0].set_xlim(start, end)
axx[0].legend()