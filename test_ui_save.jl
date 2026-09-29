using MedEye3d

# Start the event loop
MedEye3d.start_medeye3d_events()

fig = MedEye3d.Figure()
res = MedEye3d.create_metadata_window(fig)
display(fig)

sleep(1)
# we can trigger a button click!
btn = res.obs_new_lesion # No, btn_add_anat_rel is NOT exported!

# Wait! Is it exported?
