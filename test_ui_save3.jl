using MedEye3d

MedEye3d.start_medeye3d_events()
fig = MedEye3d.Figure()
res = MedEye3d.create_metadata_window(fig)

# Force anatomy to active
res._lmw_observables[:obs_new_lesion][] = 1
sleep(0.5)

# Wait... I can just call the save directly?
