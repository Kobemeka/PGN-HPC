from molecule import *
import glob
from writeMolFile import MolFileWriter

acceptor_path = "./acceptors"
output_path = "./output"
graphene_mol_file = "./graphene.mol"


show_settings = {
    "show_center_number":False,
    "show_center_distance":False,
    "show_center_distance_text":False,
    "show_atom_number":False,
    "show_ring":True,
    "show_ring_atoms":False,
    "show_ring_center":False,
    "show_ring_center_of_mass":True,
    "expected_distance": 1.42,
    "tolerance": 1e-2
}
graphene_settings = {
    "show_center_number":False,
    "show_center_distance":False,
    "show_center_distance_text":False,
    "show_atom_number":False,
    "show_ring":True,
    "show_ring_atoms":True,
    "show_ring_center":False,
    "show_ring_center_of_mass":False,
    "expected_distance": 1.42,
    "tolerance": 1e-2
}


gf = MolFile(f"{graphene_mol_file}",**graphene_settings)
graphene = Molecule(gf.atoms,gf.rings,gf.center_of_mass,gf.bonds_ids,True)
allMolFiles = [f for f in glob.glob(f"{acceptor_path}/*.mol")]

test_range = 2
test_rotation = np.pi*2
test_ratio = 25


for molfileid, molfile in enumerate(allMolFiles):
    print("="*10+"START"+"="*10)
    print(f"{molfileid+1}/{len(allMolFiles)}")
    print(molfile)

    mf = MolFile(molfile,**show_settings)

    molecule = Molecule(mf.atoms,mf.rings,mf.center_of_mass,mf.bonds_ids,True)

    gbtr = molecule.getBestTranslationRotation(graphene,test_range,test_rotation,test_range/test_ratio,test_rotation/test_ratio)

    writer = MolFileWriter([gbtr[1],graphene])
    writer.write(f"./{output_path}/{molfile.split('.')[-2].split('/')[-1]}-output-{test_ratio}.mol")
    print("="*11+"END"+"="*11+"\n\n")