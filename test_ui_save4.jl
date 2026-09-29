using MedEye3d

MedEye3d.start_medeye3d_events()
fig = MedEye3d.Figure()
res = MedEye3d.create_metadata_window(fig)

# Wait for init
sleep(1)

# Manually invoke btn_add_anat_rel (the + Row button)
# Where is it? It's inside the function. Let's just mutate anat_active_count directly!
MedEye3d.LesionMetadataWindow._all_textboxes # wait, what if I grab the observable from somewhere?
