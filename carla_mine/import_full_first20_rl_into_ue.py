"""Import the full satellite terrain with ROI-only optimization/navigation."""
import csv
import os
import unreal

ROOT=os.environ.get(
    'MINE_FULL_MAP_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs', 'm0s0_first20_rl_full_map')),
);DEST='/Game/Carla/Maps/MineFirst20RLFull';LEVEL=DEST+'/MineFirst20RLFullPreview'
tools=unreal.AssetToolsHelpers.get_asset_tools()
tex=unreal.AssetImportTask();tex.filename=ROOT+'/MineFullTexture.png';tex.destination_path=DEST;tex.destination_name='MineFirst20FullTexture';tex.automated=True;tex.replace_existing=True;tex.save=True
mesh_task=unreal.AssetImportTask();mesh_task.filename=ROOT+'/MineFirst20FullTerrain.obj';mesh_task.destination_path=DEST;mesh_task.destination_name='MineFirst20FullTerrain';mesh_task.automated=True;mesh_task.replace_existing=True;mesh_task.save=True
opt=unreal.FbxImportUI();opt.import_mesh=True;opt.import_as_skeletal=False;opt.import_materials=False;opt.import_textures=False;opt.static_mesh_import_data.combine_meshes=True;opt.static_mesh_import_data.auto_generate_collision=False;opt.static_mesh_import_data.generate_lightmap_u_vs=False;mesh_task.options=opt
tools.import_asset_tasks([tex,mesh_task]);texture=unreal.load_asset(DEST+'/MineFirst20FullTexture');mesh=unreal.load_asset(DEST+'/MineFirst20FullTerrain')
if not texture or not mesh:raise RuntimeError('full ROI map import failed')
mat=unreal.load_asset(DEST+'/M_MineFirst20FullOrtho')
if not mat:mat=tools.create_asset('M_MineFirst20FullOrtho',DEST,unreal.Material,unreal.MaterialFactoryNew())
unreal.MaterialEditingLibrary.delete_all_material_expressions(mat);sample=unreal.MaterialEditingLibrary.create_material_expression(mat,unreal.MaterialExpressionTextureSample,-300,0);sample.set_editor_property('texture',texture);unreal.MaterialEditingLibrary.connect_material_property(sample,'RGB',unreal.MaterialProperty.MP_BASE_COLOR)
rough=unreal.MaterialEditingLibrary.create_material_expression(mat,unreal.MaterialExpressionConstant,-300,120);rough.set_editor_property('r',.92);unreal.MaterialEditingLibrary.connect_material_property(rough,'',unreal.MaterialProperty.MP_ROUGHNESS);unreal.MaterialEditingLibrary.recompile_material(mat)
mesh.set_material(0,mat);mesh.get_editor_property('body_setup').set_editor_property('collision_trace_flag',unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE);unreal.EditorAssetLibrary.save_loaded_asset(mesh,False)
if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
 unreal.EditorLevelLibrary.load_level(LEVEL)
 for a in unreal.EditorLevelLibrary.get_all_level_actors():unreal.EditorLevelLibrary.destroy_actor(a)
else:unreal.EditorLevelLibrary.new_level(LEVEL)
a=unreal.EditorLevelLibrary.spawn_actor_from_object(mesh,unreal.Vector());a.set_actor_label('MineFirst20FullTerrain_Satellite_ROI_LiDAR')
sun=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight,unreal.Vector(0,0,100000),unreal.Rotator(-90,-90,90));sun.set_actor_label('MineFirst20FullSun');sc=sun.get_component_by_class(unreal.DirectionalLightComponent);sc.set_mobility(unreal.ComponentMobility.MOVABLE);sc.set_editor_property('intensity',5.0);sc.set_editor_property('cast_shadows',True)
sky=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight,unreal.Vector(0,0,100000));sky.set_actor_label('MineFirst20FullSky');sky.get_component_by_class(unreal.SkyLightComponent).set_mobility(unreal.ComponentMobility.MOVABLE)
with open(ROOT+'/navigation/terminal_points.csv') as f:
 for r in csv.DictReader(f):
  p=unreal.Vector(float(r['carla_x_m'])*100,-float(r['carla_y_m'])*100,float(r['z_relative_m'])*100);t=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.TargetPoint,p);t.set_actor_label('RLTerminal_'+r['id'])
unreal.EditorLevelLibrary.save_current_level();unreal.EditorAssetLibrary.save_directory(DEST,False,True);unreal.log('MINE_FIRST20_FULL_IMPORT_SUCCESS')
