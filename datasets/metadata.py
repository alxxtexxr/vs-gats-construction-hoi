"""
Created on Oct 02, 2017
@author: Siyuan Qi
Description of the file.
"""

# 8 classes
# hico_classes = ['__background__',  # always index 0
#                 "worker", "rebar_local", "formwork_local", "concrete_local", "rebar_global", "formwork_global", "concrete_global"]

# coco_classes = ['__background__',  # always index 0
#                 "worker", "rebar_local", "formwork_local", "concrete_local", "rebar_global", "formwork_global", "concrete_global"]

# coco_classes_pytorch = ['__background__',  # always index 0
#                         "worker", "rebar_local", "formwork_local", "concrete_local", "rebar_global", "formwork_global", "concrete_global"]

fasterrcnn_classes = [
    # background
    '__background__', # 0

    # human
    'human_engineer',           # 1 
    'human_helper',             # 2 
    'human_master',             # 3 
    'human_operator',           # 4 
    'human_safety_officer',     # 5 
    'human_squating_engineer',  # 6 
    'human_squating_helper',    # 7 
    'human_squating_master',    # 8 
    'human_squating_welder',    # 9 
    'human_squating_worker',    # 10
    'human_supervisor',         # 11
    'human_welder',             # 12
    'human_worker',             # 13

    # equipemnt
    'equipment_bucket',                     # 14
    'equipment_carried_concrete_leveler',   # 15
    'equipment_carried_drawing',            # 16
    'equipment_carried_general_tools',      # 17
    'equipment_carried_hammer',             # 18
    'equipment_carried_screwdriver',        # 19
    'equipment_carried_wire_cutter',        # 20
    'equipment_scaffolding',                # 21
    'equipment_unused_bag',                 # 22
    'equipment_unused_bucket',              # 23
    'equipment_unused_cable_roller',        # 24
    'equipment_unused_car',                 # 25
    'equipment_unused_cart',                # 26
    'equipment_unused_chair',               # 27
    'equipment_unused_cleaning_tools',      # 28
    'equipment_unused_concrete_finisher',   # 29
    'equipment_unused_concrete_leveler',    # 30
    'equipment_unused_cutting_machine',     # 31
    'equipment_unused_cutting_table',       # 32
    'equipment_unused_drawing',             # 33
    'equipment_unused_excavator',           # 34
    'equipment_unused_fan',                 # 35
    'equipment_unused_gas_tank',            # 36
    'equipment_unused_general_tools',       # 37
    'equipment_unused_hammer',              # 38
    'equipment_unused_hose',                # 39
    'equipment_unused_ladder',              # 40
    'equipment_unused_measuring_tape',      # 41
    'equipment_unused_mobile_crane',        # 42
    'equipment_unused_rebar_machine',       # 43
    'equipment_unused_rubber_cone',         # 44
    'equipment_unused_scaffolding',         # 45
    'equipment_unused_theodolite',          # 46
    'equipment_unused_truck',               # 47
    'equipment_unused_welding_machine',     # 48
    'equipment_used_bucket',                # 49
    'equipment_used_cart',                  # 50
    'equipment_used_cleaning_tools',        # 51
    'equipment_used_concrete_finisher',     # 52
    'equipment_used_concrete_leveler',      # 53
    'equipment_used_concrete_pump',         # 54
    'equipment_used_concrete_vibrator',     # 55
    'equipment_used_cutting_table',         # 56
    'equipment_used_drawing',               # 57
    'equipment_used_general_tools',         # 58
    'equipment_used_hammer',                # 59
    'equipment_used_ladder',                # 60
    'equipment_used_measuring_tape',        # 61
    'equipment_used_mobile_crane',          # 62
    'equipment_used_nail',                  # 63
    'equipment_used_rebar_machine',         # 64
    'equipment_used_theodolite',            # 65
    'equipment_used_welding_machine',       # 66

    # material
    'material_column_rebar_global_25',                      # 67
    'material_column_rebar_local_25',                       # 68
    'material_column_rebar_local_50',                       # 69
    'material_concrete_global_100',                         # 70
    'material_concrete_global_25',                          # 71
    'equipment_used_nailing_machine',       # 72 (equipment)
    'material_concrete_global_50',                          # 73 (material)
    'equipment_used_scaffolding',           # 74 (equipment)
    'material_concrete_global_75',                          # 75
    'material_concrete_local_100',                          # 76
    'material_concrete_local_50',                           # 77
    'material_concrete_local_75',                           # 78
    'material_finished_plywood_formwork_global_100',        # 79
    'material_finished_plywood_formwork_global_25',         # 80
    'material_finished_plywood_formwork_global_75',         # 81
    'material_finished_plywood_formwork_local_75',          # 82
    'material_finished_steel_formwork_global_25',           # 83
    'material_finished_steel_formwork_global_75',           # 84
    'material_finished_steel_formwork_local_100',           # 85
    'material_finished_steel_formwork_local_25',            # 86
    'material_rebar_rod_local_0',                           # 87
    'material_slab_rebar_global_100',                       # 88
    'material_slab_rebar_global_25',                        # 89
    'material_slab_rebar_global_50',                        # 90
    'material_slab_rebar_global_75',                        # 91
    'material_slab_rebar_local_100',                        # 92
    'material_slab_rebar_local_25',                         # 93
    'material_slab_rebar_local_50',                         # 94
    'material_slab_rebar_local_75',                         # 95
    'material_unfinished_plywood_formwork_local_0',         # 96
    'material_unfinished_styrofoam_formwork_local_0',       # 97
    'material_unused_column_rebar_local_25',                # 98
    'material_unused_column_rebar_local_25-',               # 99
    'material_unused_column_rebar_local_50',                # 100
    'material_unused_concrete_local_100',                   # 101
    'material_unused_concrete_local_25',                    # 102
    'material_unused_concrete_local_50',                    # 103
    'material_unused_finished_plywood_formwork_local_50',   # 104
    'material_unused_finished_plywood_formwork_local_75',   # 105
    'material_unused_finished_steel_formwork_local_75',     # 106
    'material_unused_plywood_formwork_local_0',             # 107
    'material_unused_plywood_formwork_local_75',            # 108
    'material_unused_rebar_column_local_25',                # 109
    'material_unused_rebar_rod_local_0',                    # 110
    'material_unused_slab_rebar_global_0',                  # 111
    'material_unused_slab_rebar_global_100',                # 112
    'material_unused_slab_rebar_local_100',                 # 113
    'material_unused_slab_rebar_local_25',                  # 114
    'material_unused_slab_rebar_local_75',                  # 115
    'material_unused_unfinished_plywood_formwork_local_0',  # 116
    'material_used_column_rebar_global_25',                 # 117
    'material_used_column_rebar_local_25',                  # 118
    'material_used_column_rebar_local_50',                  # 119
    'material_used_column_rebar_local_75',                  # 120
    'material_used_concrete_global_0',                      # 121
    'material_used_concrete_global_100',                    # 122
    'material_used_concrete_global_25',                     # 123
    'material_used_concrete_global_50',                     # 124
    'material_used_concrete_local_0',                       # 125
    'material_used_concrete_local_100',                     # 126
    'material_used_concrete_local_25',                      # 127
    'material_used_concrete_local_50',                      # 128
    'material_used_concrete_local_75',                      # 129
    'material_used_finished_plywood_formwork_global_100',   # 130
    'material_used_finished_plywood_formwork_global_25',    # 131
    'material_used_finished_plywood_formwork_global_50',    # 132
    'material_used_finished_plywood_formwork_global_75',    # 133
    'material_used_finished_plywood_formwork_local_100',    # 134
    'material_used_finished_plywood_formwork_local_25',     # 135
    'material_used_finished_plywood_formwork_local_50',     # 136
    'material_used_finished_plywood_formwork_local_75',     # 137
    'material_used_finished_steel_formwork_global_25',      # 138
    'material_used_finished_steel_formwork_global_75',      # 139
    'material_used_finished_steel_formwork_local_100',      # 140
    'material_used_finished_steel_formwork_local_50',       # 141
    'material_used_finished_steel_formwork_local_75',       # 142
    'material_used_plywood_formwork_global_100',            # 143
    'material_used_plywood_formwork_global_25',             # 144
    'material_used_rebar_rod_global_0',                     # 145
    'material_used_rebar_rod_local_0',                      # 146
    'material_used_slab_rebar_global_0',                    # 147
    'material_used_slab_rebar_global_100',                  # 148
    'material_used_slab_rebar_global_25',                   # 149
    'material_used_slab_rebar_global_50',                   # 150
    'material_used_slab_rebar_local_100',                   # 151
    'material_used_slab_rebar_local_25',                    # 152
    'material_used_slab_rebar_local_50',                    # 153
    'material_used_slab_rebar_local_75',                    # 154
    'material_used_steel_formwork_global_100',              # 155
    'material_used_unfinished_plywood_formwork_global_0',   # 156
    'material_used_unfinished_plywood_formwork_local_0',    # 157
    'material_used_unfinished_plywood_formwork_local_50',   # 158
    'material_used_unfinished_steel_formwork_local_0',      # 159
    'material_used_unfinished_styrofoam_formwork_local_0',  # 160
]
hico_classes = [
    '__background__', # Background class must always at index 0
    'human',          # Human class must always at index 1
    'equipment', 
    # 'material',
    'rebar',
    'formwork',
    'concrete',
]
# hico_classes = fasterrcnn_classes
coco_classes = hico_classes
coco_classes_pytorch= hico_classes

# Precompute a mapping of fasterrcnn_label -> hico_label
fasterrcnn_to_hico_label_map = {}
for hico_label, hico_class in enumerate(hico_classes):
    for fasterrcnn_label, fasterrcnn_class in enumerate(fasterrcnn_classes):
        if hico_class in fasterrcnn_class:
            fasterrcnn_to_hico_label_map[fasterrcnn_label] = hico_label

def fasterrcnn_to_hico_label(fasterrcnn_label):
    return fasterrcnn_to_hico_label_map.get(fasterrcnn_label, -1)

coco_to_hico = ['N/A' if c not in hico_classes else hico_classes.index(c) for c in coco_classes_pytorch]
coco_pytorch_to_coco = ['N/A' if c not in coco_classes else coco_classes.index(c) for c in coco_classes_pytorch]
hico_to_coco = [coco_classes.index(c) for c in hico_classes]

hoi_classes = [
    # human
    'supervise',        # 0
    'collaborate_with', # 1
    'assist',           # 2
    'lead',             # 3
    'coordinate_with',  # 4
    'listen_to',        # 5

    # rebar
    'tie',              # 6
    'erect',            # 7
    'prepare_rebar',    # 8
    'transport',        # 9

    # formwork
    'install',          # 10
    'prepare_formwork', # 11

    # concrete
    'pour',             # 12
    'finish',           # 13

    # equipment         
    'use',              # 14
    'carry',            # 15

    # all
    'inspect',          # 16
    'no_interaction',   # 17
]
assert len(set(hoi_classes)) == len(hoi_classes), "HOI classes must be unique"
action_classes = hoi_classes

# Used in metadata.py, hico_train_val_test_data.py, vis_tool.py
hoi_to_action = [action_classes.index(c) for c in hoi_classes]

# Used in metadata.py and hico_eval.py
obj_to_hoi = [
    [],                         # human-background
    [0, 1, 2, 3, 4, 5, 16, 17], # human-human
    [14, 15, 16, 17],           # human-equipment
    [6, 7, 8, 9, 16, 17],       # human-rebar
    [10, 11, 16, 17],           # human-formwork
    [12, 13, 16, 17],           # human-concrete
]
assert len(obj_to_hoi) == len(hico_classes), "obj_to_hoi length must be equal to hico_classes length"

# obj_hoi_index_fasterrcnn = [
#     # human-background -> no interaction allowed
#     (), # 0 - human-background

#     # human-human -> all interactions allowed
#     # NOTE: 
#     # For temporary, all human interaction are not conditioned, 
#     # which means all human interactions are not bound with the role of the humans
#     (1, len(hoi_classes)), # 1 - human-human_engineer
#     (1, len(hoi_classes)), # 2 - human-human_helper
#     (1, len(hoi_classes)), # 3 - human-human_master
#     (1, len(hoi_classes)), # 4 - human-human_operator
#     (1, len(hoi_classes)), # 5 - human-human_safety_officer
#     (1, len(hoi_classes)), # 6 - human-human_squating_engineer
#     (1, len(hoi_classes)), # 7 - human-human_squating_helper
#     (1, len(hoi_classes)), # 8 - human-human_squating_master
#     (1, len(hoi_classes)), # 9 - human-human_squating_welder
#     (1, len(hoi_classes)), # 10 - human-human_squating_worker
#     (1, len(hoi_classes)), # 11 - human-human_supervisor
#     (1, len(hoi_classes)), # 12 - human-human_welder
#     (1, len(hoi_classes)), # 13 - human-human_worker

#     # human-equipemnt -> no interaction allowed
#     (0, 0), # 14 - human-equipment_bucket 
#     (0, 0), # 15 - human-equipment_carried_concrete_leveler 
#     (0, 0), # 16 - human-equipment_carried_drawing 
#     (0, 0), # 17 - human-equipment_carried_general_tools 
#     (0, 0), # 18 - human-equipment_carried_hammer 
#     (0, 0), # 19 - human-equipment_carried_screwdriver 
#     (0, 0), # 20 - human-equipment_carried_wire_cutter 
#     (0, 0), # 21 - human-equipment_scaffolding 
#     (0, 0), # 22 - human-equipment_unused_bag 
#     (0, 0), # 23 - human-equipment_unused_bucket 
#     (0, 0), # 24 - human-equipment_unused_cable_roller 
#     (0, 0), # 25 - human-equipment_unused_car 
#     (0, 0), # 26 - human-equipment_unused_cart 
#     (0, 0), # 27 - human-equipment_unused_chair 
#     (0, 0), # 28 - human-equipment_unused_cleaning_tools 
#     (0, 0), # 29 - human-equipment_unused_concrete_finisher 
#     (0, 0), # 30 - human-equipment_unused_concrete_leveler 
#     (0, 0), # 31 - human-equipment_unused_cutting_machine 
#     (0, 0), # 32 - human-equipment_unused_cutting_table 
#     (0, 0), # 33 - human-equipment_unused_drawing 
#     (0, 0), # 34 - human-equipment_unused_excavator 
#     (0, 0), # 35 - human-equipment_unused_fan 
#     (0, 0), # 36 - human-equipment_unused_gas_tank 
#     (0, 0), # 37 - human-equipment_unused_general_tools 
#     (0, 0), # 38 - human-equipment_unused_hammer 
#     (0, 0), # 39 - human-equipment_unused_hose 
#     (0, 0), # 40 - human-equipment_unused_ladder 
#     (0, 0), # 41 - human-equipment_unused_measuring_tape 
#     (0, 0), # 42 - human-equipment_unused_mobile_crane 
#     (0, 0), # 43 - human-equipment_unused_rebar_machine 
#     (0, 0), # 44 - human-equipment_unused_rubber_cone 
#     (0, 0), # 45 - human-equipment_unused_scaffolding 
#     (0, 0), # 46 - human-equipment_unused_theodolite 
#     (0, 0), # 47 - human-equipment_unused_truck 
#     (0, 0), # 48 - human-equipment_unused_welding_machine 
#     (0, 0), # 49 - human-equipment_used_bucket 
#     (0, 0), # 50 - human-equipment_used_cart 
#     (0, 0), # 51 - human-equipment_used_cleaning_tools 
#     (0, 0), # 52 - human-equipment_used_concrete_finisher 
#     (0, 0), # 53 - human-equipment_used_concrete_leveler 
#     (0, 0), # 54 - human-equipment_used_concrete_pump 
#     (0, 0), # 55 - human-equipment_used_concrete_vibrator 
#     (0, 0), # 56 - human-equipment_used_cutting_table 
#     (0, 0), # 57 - human-equipment_used_drawing 
#     (0, 0), # 58 - human-equipment_used_general_tools 
#     (0, 0), # 59 - human-equipment_used_hammer 
#     (0, 0), # 60 - human-equipment_used_ladder 
#     (0, 0), # 61 - human-equipment_used_measuring_tape 
#     (0, 0), # 62 - human-equipment_used_mobile_crane 
#     (0, 0), # 63 - human-equipment_used_nail 
#     (0, 0), # 64 - human-equipment_used_rebar_machine 
#     (0, 0), # 65 - human-equipment_used_theodolite 
#     (0, 0), # 66 - human-equipment_used_welding_machine 

#     # material -> no interaction allowed
#     (0, 0), # 67 - human-material_column_rebar_global_25 
#     (0, 0), # 68 - human-material_column_rebar_local_25 
#     (0, 0), # 69 - human-material_column_rebar_local_50 
#     (0, 0), # 70 - human-material_concrete_global_100 
#     (0, 0), # 71 - human-material_concrete_global_25 
#     (0, 0), # 72 - human-equipment_used_nailing_machine (human-equipment)
#     (0, 0), # 73 - human-material_concrete_global_50 (human-material)
#     (0, 0), # 74 - human-equipment_used_scaffolding (human-equipemnt)
#     (0, 0), # 75 - human-material_concrete_global_75 
#     (0, 0), # 76 - human-material_concrete_local_100 
#     (0, 0), # 77 - human-material_concrete_local_50 
#     (0, 0), # 78 - human-material_concrete_local_75 
#     (0, 0), # 79 - human-material_finished_plywood_formwork_global_100 
#     (0, 0), # 80 - human-material_finished_plywood_formwork_global_25 
#     (0, 0), # 81 - human-material_finished_plywood_formwork_global_75 
#     (0, 0), # 82 - human-material_finished_plywood_formwork_local_75 
#     (0, 0), # 83 - human-material_finished_steel_formwork_global_25 
#     (0, 0), # 84 - human-material_finished_steel_formwork_global_75 
#     (0, 0), # 85 - human-material_finished_steel_formwork_local_100 
#     (0, 0), # 86 - human-material_finished_steel_formwork_local_25 
#     (0, 0), # 87 - human-material_rebar_rod_local_0 
#     (0, 0), # 88 - human-material_slab_rebar_global_100 
#     (0, 0), # 89 - human-material_slab_rebar_global_25 
#     (0, 0), # 90 - human-material_slab_rebar_global_50 
#     (0, 0), # 91 - human-material_slab_rebar_global_75 
#     (0, 0), # 92 - human-material_slab_rebar_local_100 
#     (0, 0), # 93 - human-material_slab_rebar_local_25 
#     (0, 0), # 94 - human-material_slab_rebar_local_50 
#     (0, 0), # 95 - human-material_slab_rebar_local_75 
#     (0, 0), # 96 - human-material_unfinished_plywood_formwork_local_0 
#     (0, 0), # 97 - human-material_unfinished_styrofoam_formwork_local_0 
#     (0, 0), # 98 - human-material_unused_column_rebar_local_25 
#     (0, 0), # 99 - human-material_unused_column_rebar_local_25- 
#     (0, 0), # 100 - human-material_unused_column_rebar_local_50 
#     (0, 0), # 101 - human-material_unused_concrete_local_100 
#     (0, 0), # 102 - human-material_unused_concrete_local_25 
#     (0, 0), # 103 - human-material_unused_concrete_local_50 
#     (0, 0), # 104 - human-material_unused_finished_plywood_formwork_local_50 
#     (0, 0), # 105 - human-material_unused_finished_plywood_formwork_local_75 
#     (0, 0), # 106 - human-material_unused_finished_steel_formwork_local_75 
#     (0, 0), # 107 - human-material_unused_plywood_formwork_local_0 
#     (0, 0), # 108 - human-material_unused_plywood_formwork_local_75 
#     (0, 0), # 109 - human-material_unused_rebar_column_local_25 
#     (0, 0), # 110 - human-material_unused_rebar_rod_local_0 
#     (0, 0), # 111 - human-material_unused_slab_rebar_global_0 
#     (0, 0), # 112 - human-material_unused_slab_rebar_global_100 
#     (0, 0), # 113 - human-material_unused_slab_rebar_local_100 
#     (0, 0), # 114 - human-material_unused_slab_rebar_local_25 
#     (0, 0), # 115 - human-material_unused_slab_rebar_local_75 
#     (0, 0), # 116 - human-material_unused_unfinished_plywood_formwork_local_0 
#     (0, 0), # 117 - human-material_used_column_rebar_global_25 
#     (0, 0), # 118 - human-material_used_column_rebar_local_25 
#     (0, 0), # 119 - human-material_used_column_rebar_local_50 
#     (0, 0), # 120 - human-material_used_column_rebar_local_75 
#     (0, 0), # 121 - human-material_used_concrete_global_0 
#     (0, 0), # 122 - human-material_used_concrete_global_100 
#     (0, 0), # 123 - human-material_used_concrete_global_25 
#     (0, 0), # 124 - human-material_used_concrete_global_50 
#     (0, 0), # 125 - human-material_used_concrete_local_0 
#     (0, 0), # 126 - human-material_used_concrete_local_100 
#     (0, 0), # 127 - human-material_used_concrete_local_25 
#     (0, 0), # 128 - human-material_used_concrete_local_50 
#     (0, 0), # 129 - human-material_used_concrete_local_75 
#     (0, 0), # 130 - human-material_used_finished_plywood_formwork_global_100 
#     (0, 0), # 131 - human-material_used_finished_plywood_formwork_global_25 
#     (0, 0), # 132 - human-material_used_finished_plywood_formwork_global_50 
#     (0, 0), # 133 - human-material_used_finished_plywood_formwork_global_75 
#     (0, 0), # 134 - human-material_used_finished_plywood_formwork_local_100 
#     (0, 0), # 135 - human-material_used_finished_plywood_formwork_local_25 
#     (0, 0), # 136 - human-material_used_finished_plywood_formwork_local_50 
#     (0, 0), # 137 - human-material_used_finished_plywood_formwork_local_75 
#     (0, 0), # 138 - human-material_used_finished_steel_formwork_global_25 
#     (0, 0), # 139 - human-material_used_finished_steel_formwork_global_75 
#     (0, 0), # 140 - human-material_used_finished_steel_formwork_local_100 
#     (0, 0), # 141 - human-material_used_finished_steel_formwork_local_50 
#     (0, 0), # 142 - human-material_used_finished_steel_formwork_local_75 
#     (0, 0), # 143 - human-material_used_plywood_formwork_global_100 
#     (0, 0), # 144 - human-material_used_plywood_formwork_global_25 
#     (0, 0), # 145 - human-material_used_rebar_rod_global_0 
#     (0, 0), # 146 - human-material_used_rebar_rod_local_0 
#     (0, 0), # 147 - human-material_used_slab_rebar_global_0 
#     (0, 0), # 148 - human-material_used_slab_rebar_global_100 
#     (0, 0), # 149 - human-material_used_slab_rebar_global_25 
#     (0, 0), # 150 - human-material_used_slab_rebar_global_50 
#     (0, 0), # 151 - human-material_used_slab_rebar_local_100 
#     (0, 0), # 152 - human-material_used_slab_rebar_local_25 
#     (0, 0), # 153 - human-material_used_slab_rebar_local_50 
#     (0, 0), # 154 - human-material_used_slab_rebar_local_75 
#     (0, 0), # 155 - human-material_used_steel_formwork_global_100 
#     (0, 0), # 156 - human-material_used_unfinished_plywood_formwork_global_0 
#     (0, 0), # 157 - human-material_used_unfinished_plywood_formwork_local_0 
#     (0, 0), # 158 - human-material_used_unfinished_plywood_formwork_local_50 
#     (0, 0), # 159 - human-material_used_unfinished_steel_formwork_local_0 
#     (0, 0), # 160 - human-material_used_unfinished_styrofoam_formwork_local_0 
# ]

# Used in vis_tool.py and metadata.py
# obj_to_hoi = [hoi_classes[x[0] - 1: x[1]] for x in obj_hoi_index]
# obj_to_hoi_fasterrcnn = [hoi_classes[x[0] - 1: x[1]] for x in obj_hoi_index_fasterrcnn] 

# obj_actions = [[action_classes.index(y) for y in x] for x in obj_to_hoi] # only used in metadata.py

# CUSTOM
# import pandas as pd

# def get_hhi_actions_map(hhi_action_map_path='datasets/hhi_actions_map.csv', start_row=1, start_col=1):
#     df = pd.read_csv(hhi_action_map_path, header=None)
#     df = df.iloc[start_row:, start_col:]
#     actions_map = df.applymap(lambda x: list(map(int, str(x).split(',')))).values
#     return actions_map

# hhi_actions_map = get_hhi_actions_map()

# def filter_irrelevant_hhi_actions(h_src_label, h_dst_label, action_labels, label_type='fasterrcnn'):
#     # TODO: Implement handling for the HICO label type

#     # Find the position of the first class label containing 'human'
#     classes = fasterrcnn_classes if label_type == 'fasterrcnn' else hico_classes
#     first_human_idx = next(i for i, class_ in enumerate(classes) if 'human' in class_)

#     # Normalize the human labels to be within the range: 0 - number of human labels (13 in this case)
#     h_src_label -= first_human_idx
#     h_dst_label -= first_human_idx

#     # Filter irrelevant actions based on the HHI actions map
#     relevant_action_labels = hhi_actions_map[h_src_label, h_dst_label]
#     filtered_action_labels = [action_label for action_label in action_labels if action_label in relevant_action_labels]
#     return filtered_action_labels

# only used in metadata.py
# def action_to_obj_idx(obj_class, action_hico):
#     action_coco = action_classes[action_hico]
#     obj_interval = obj_hoi_index[obj_class]
#     hois = hoi_classes[obj_interval[0] - 1 : obj_interval[1]]
#     return hois.index(action_coco)

# def main():
#     pass

# if __name__ == '__main__':
#     main()