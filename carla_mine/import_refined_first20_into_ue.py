"""Add the 1 m LiDAR-refined ROI surface to the full satellite CARLA map."""
import csv
import os
import unreal

ROOT=os.environ.get(
    'MINE_V2_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'v2')),
)
DEST='/Game/Carla/Maps/MineFirst20RLFull'
LEVEL=DEST+'/MineFirst20RLFullRefinedV2'
tools=unreal.AssetToolsHelpers.get_asset_tools()

task=unreal.AssetImportTask();task.filename=ROOT+'/MineFirst20RefinedPatch.obj';task.destination_path=DEST
task.destination_name='MineFirst20RefinedPatchV2';task.automated=True;task.replace_existing=True;task.save=True
opt=unreal.FbxImportUI();opt.import_mesh=True;opt.import_as_skeletal=False;opt.import_materials=False
opt.import_textures=False;opt.static_mesh_import_data.combine_meshes=True
opt.static_mesh_import_data.auto_generate_collision=False;opt.static_mesh_import_data.generate_lightmap_u_vs=False
task.options=opt;tools.import_asset_tasks([task])
base=unreal.load_asset(DEST+'/MineFirst20FullTerrain');patch=unreal.load_asset(DEST+'/MineFirst20RefinedPatchV2')
mat=unreal.load_asset(DEST+'/M_MineFirst20FullOrtho')
if not base or not patch or not mat:raise RuntimeError('required full-map/refined assets are missing')
patch.set_material(0,mat)
patch.get_editor_property('body_setup').set_editor_property('collision_trace_flag',unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
unreal.EditorAssetLibrary.save_loaded_asset(patch,False)

if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
 unreal.EditorLevelLibrary.load_level(LEVEL)
 for a in unreal.EditorLevelLibrary.get_all_level_actors():unreal.EditorLevelLibrary.destroy_actor(a)
else:unreal.EditorLevelLibrary.new_level(LEVEL)

visual=unreal.EditorLevelLibrary.spawn_actor_from_object(base,unreal.Vector());visual.set_actor_label('MineFullSatelliteTerrain_Visual')
vc=visual.get_component_by_class(unreal.StaticMeshComponent);vc.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
surface=unreal.EditorLevelLibrary.spawn_actor_from_object(patch,unreal.Vector());surface.set_actor_label('MineFirst20_LiDARRefined_1m_Collision')
pc=surface.get_component_by_class(unreal.StaticMeshComponent);pc.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)

sun=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight,unreal.Vector(0,0,100000),unreal.Rotator(-90,-90,90))
sun.set_actor_label('MineHybridSun');sc=sun.get_component_by_class(unreal.DirectionalLightComponent)
sc.set_mobility(unreal.ComponentMobility.MOVABLE);sc.set_editor_property('intensity',5.0);sc.set_editor_property('cast_shadows',True)
sky=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight,unreal.Vector(0,0,100000));sky.set_actor_label('MineHybridSky')
sky.get_component_by_class(unreal.SkyLightComponent).set_mobility(unreal.ComponentMobility.MOVABLE)
with open(ROOT+'/navigation/terminal_points.csv') as f:
 for r in csv.DictReader(f):
  p=unreal.Vector(float(r['carla_x_m'])*100,-float(r['carla_y_m'])*100,float(r['z_relative_m'])*100)
  t=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.TargetPoint,p);t.set_actor_label('RLTerminal_'+r['id'])
unreal.EditorLevelLibrary.save_current_level();unreal.EditorAssetLibrary.save_directory(DEST,False,True)
unreal.log('MINE_FIRST20_LIDAR_REFINED_IMPORT_SUCCESS')
