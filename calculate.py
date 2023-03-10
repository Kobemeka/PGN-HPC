from molecule import *
import argparse
import matplotlib.pyplot as plt
import glob

acceptor_path = "./acceptors"
output_path = "./output"
graphene_mol_file = "./graphene.mol"
draw_graphene = False

allMolFiles = [f for f in glob.glob(f"{acceptor_path}/*.mol")]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument("--center_number",default=False)
    parser.add_argument("--center_distance",default=False)
    parser.add_argument("--center_distance_text",default=False)
    parser.add_argument("--atom_number",default=False)
    parser.add_argument("--ring",default=True)
    parser.add_argument("--ring_atoms",default=True)
    parser.add_argument("--ring_center",default=True)
    parser.add_argument("--ring_center_of_mass",default=True)
    parser.add_argument("--dump_center_info",default=False)
    parser.add_argument("--draw_file",default=True)


    args = parser.parse_args()

    show_settings = {
        "show_center_number":args.center_number,
        "show_center_distance":args.center_distance,
        "show_center_distance_text":args.center_distance_text,
        "show_atom_number":args.atom_number,
        "show_ring":args.ring,
        "show_ring_atoms":args.ring_atoms,
        "show_ring_center":args.ring_center,
        "show_ring_center_of_mass":args.ring_center_of_mass,

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
    }

    for file in allMolFiles:
        print(file)
        mf = MolFile(file,**show_settings)
        gf = MolFile(f"{graphene_mol_file}",**graphene_settings)
        molecule = Molecule(mf.atoms,mf.rings,mf.center_of_mass,mf.bonds_ids)
        graphene = Molecule(gf.atoms,gf.rings,gf.center_of_mass,gf.bonds_ids)


        fig = plt.figure(figsize=(12,12),dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel("X - AXIS")
        ax.set_ylabel("Y - AXIS")
        ax.set_zlabel("Z - AXIS")

        test_range = 2
        test_rotation = np.pi*2
        test_ratio = 25

        
        gbtr = molecule.getBestTranslationRotation(graphene,test_range,test_rotation,test_range/test_ratio,test_rotation/test_ratio)
        with open(f"{output_path}/output.txt","a") as outf:
            outf.write(f"{'-'*10}{file.replace(f'{acceptor_path}/','')}{'-'*10}\n\n")
            outf.write(f"Translation Range\tRotation Range\tStep\n")
            outf.write(f"{'_'*20}\n")
            outf.write(f"{test_range}(A)\t{test_rotation}(rad)\t{test_ratio}\n\n")
            outf.write(f"Center\tGr C Distance\n")
            outf.write(f"{'_'*40}\n")

        gbtr[1].CADoutput(ax,graphene,False)
        gbtr[1].CenterCombOutput()
        
        if draw_graphene:
            graphene.draw(ax)

        gbtr[1].draw(ax)
        plt.title(file)
        ax.view_init(azim=0, elev=90)
        
        plt.savefig(f"{output_path}/{file.replace(f'{acceptor_path}/','')}.png",dpi=120)