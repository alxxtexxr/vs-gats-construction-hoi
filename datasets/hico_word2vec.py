import gensim
from datasets.metadata import coco_classes, fasterrcnn_classes
from datasets.hico_constants import HicoConstants
import os
import h5py
from tqdm import tqdm

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('datasets/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

data_const = HicoConstants()
original_keys = list(model.key_to_index.keys())
upper_keys = [str.upper(x) for x in original_keys]
hico_word2vec = os.path.join(data_const.proc_dir,'hico_word2vec.hdf5')
file = h5py.File(hico_word2vec, 'w')
# for name in fasterrcnn_classes:
for name in coco_classes + fasterrcnn_classes:
    if name == '__background__': 
        continue

    # human
    elif 'engineer' in name:
        index = upper_keys.index(str.upper('engineer'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'helper' in name:
        index = upper_keys.index(str.upper('helper')) # *
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'master' in name:
        index = upper_keys.index(str.upper('master')) # *
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'operater' in name:
        index = upper_keys.index(str.upper('operater'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'officer' in name:
        index = upper_keys.index(str.upper('officer'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'welder' in name:
        index = upper_keys.index(str.upper('welder'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'worker' in name:
        index = upper_keys.index(str.upper('worker'))
        file.create_dataset(name, data=model[original_keys[index]])

    # equipement
    elif 'bucket' in name: 
        index = upper_keys.index(str.upper('bucket'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'level' in name: 
        index = upper_keys.index(str.upper('level'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'drawing' in name: 
        index = upper_keys.index(str.upper('drawing'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'tool' in name: 
        index = upper_keys.index(str.upper('shovel'))
        file.create_dataset(name, data=model[original_keys[index]])    
    elif 'hammer' in name: 
        index = upper_keys.index(str.upper('hammer'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'screwdriver' in name: 
        index = upper_keys.index(str.upper('screwdriver'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'scaffolding' in name: 
        index = upper_keys.index(str.upper('scaffolding'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'bag' in name: 
        index = upper_keys.index(str.upper('bag'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'cable' in name: 
        index = upper_keys.index(str.upper('cable'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'car' in name: 
        index = upper_keys.index(str.upper('car'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'cart' in name: 
        index = upper_keys.index(str.upper('wheelbarrow'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'chair' in name: 
        index = upper_keys.index(str.upper('chair'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'finisher' in name: 
        index = upper_keys.index(str.upper('finisher'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'machine' in name: 
        index = upper_keys.index(str.upper('machine'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'table' in name: 
        index = upper_keys.index(str.upper('table'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'excavator' in name: 
        index = upper_keys.index(str.upper('excavator'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'fan' in name: 
        index = upper_keys.index(str.upper('fan'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'gas' in name: 
        index = upper_keys.index(str.upper('gas'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'hose' in name: 
        index = upper_keys.index(str.upper('hose'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'ladder' in name: 
        index = upper_keys.index(str.upper('ladder'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'measure' in name: 
        index = upper_keys.index(str.upper('measure'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'crane' in name: 
        index = upper_keys.index(str.upper('crane'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'cone' in name: 
        index = upper_keys.index(str.upper('cone'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'theodolite' in name: 
        index = upper_keys.index(str.upper('theodolite'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'truck' in name: 
        index = upper_keys.index(str.upper('truck'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'pump' in name: 
        index = upper_keys.index(str.upper('pump'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'vibrator' in name: 
        index = upper_keys.index(str.upper('vibrator'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'nail' in name: 
        index = upper_keys.index(str.upper('nail'))
        file.create_dataset(name, data=model[original_keys[index]])
    # elif 'shovel' in name: 
    #     index = upper_keys.index(str.upper('shovel'))
    #     file.create_dataset(name, data=model[original_keys[index]])
    # elif 'broom' in name: 
    #     index = upper_keys.index(str.upper('broom'))
    #     file.create_dataset(name, data=model[original_keys[index]])
    # elif 'checkboard' in name: 
    #     index = upper_keys.index(str.upper('checkerboard'))
    #     file.create_dataset(name, data=model[original_keys[index]])
    
    # material
    elif 'rebar' in name:
        index = upper_keys.index(str.upper('rebar'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'concrete' in name: 
        index = upper_keys.index(str.upper('concrete'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'formwork' in name: 
        index = upper_keys.index(str.upper('formwork'))
        file.create_dataset(name, data=model[original_keys[index]])
    
    # Others
    elif 'material' in name: 
        index = upper_keys.index(str.upper('material'))
        file.create_dataset(name, data=model[original_keys[index]])
    elif 'equipment' in name: 
        index = upper_keys.index(str.upper('equipment'))
        file.create_dataset(name, data=model[original_keys[index]])    
    elif 'human' in name: 
        index = upper_keys.index(str.upper('person'))
        file.create_dataset(name, data=model[original_keys[index]])
    else: 
        # index = upper_keys.index(str.upper(name))
        # file.create_dataset(name, data=model[original_keys[index]])
        raise Exception(f"Error: class name '{name}' is not exist in word2vec!")

    # if name == '__background__': continue

    # if name == 'sports_ball':
    #     index = upper_keys.index(str.upper('footballs'))
    #     file.create_dataset(name, data=model[original_keys[index]])

    # elif name == 'baseball_glove': 
    #     index = upper_keys.index(str.upper('mitt'))
    #     file.create_dataset(name, data=model[original_keys[index]])

    # elif name == 'wine_glass': 
    #     index = upper_keys.index(str.upper('wine_glasses'))
    #     file.create_dataset(name, data=model[original_keys[index]])

    # elif name == 'hair_drier': 
    #     index = upper_keys.index(str.upper('hair_driers'))
    #     file.create_dataset(name, data=model[original_keys[index]])
    # else: 
    #     index = upper_keys.index(str.upper(name))
    #     file.create_dataset(name, data=model[original_keys[index]])

file.close()
