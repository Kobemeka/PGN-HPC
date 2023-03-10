def findMaxLength(coord):
    return max([len(str(c).split(".")[0]) for c in coord])

class MolFileWriter:
    def __init__(self,molecules) -> None:
        self.molecules = molecules
    
    def write(self,file):
        title = "     RDKit          2D"
        total_atoms = sum([len(a.atoms) for a in self.molecules])
        total_bonds = sum([len(b.bonds_ids) for b in self.molecules])
        obs1 = "0  0"
        chiral = 0 # False
        obs2 = "0  0  0  0  0999"
        fileFormat = "V2000"


        # atoms block
        otherAttrs = (" 0  "*12).strip()
        with open(file,"w") as f:
            f.write("\n")
            f.write(f"{title}\n")
            f.write("\n")
            f.write(f"{(3 - len(str(total_atoms)))*' '}{total_atoms}{(3 - len(str(total_bonds)))*' '}{total_bonds}  {obs1}  {chiral}  {obs2} {fileFormat}\n")

            # atoms
            for molecule in self.molecules:
                for atom in molecule.atoms:
                    cx = f"{atom.coordinate.x:.4f}"
                    cy = f"{atom.coordinate.y:.4f}"
                    cz = f"{atom.coordinate.z:.4f}"
                    f.write(f"{(10 - len(cx))*' '}{cx}{(10 - len(cy))*' '}{cy}{(10 - len(cz))*' '}{cz} {str(atom.symbol)}{(3-len(str(atom.symbol)))*' '} {otherAttrs}\n")
                

                # bonds
            bstartindex = 0
            for molecule in self.molecules:
                for bond in molecule.bonds_ids:
                    fb = bstartindex + bond[0] + 1
                    sb = bstartindex + bond[1] + 1
                    f.write(f"{(3-len(str(fb)))*' '}{fb}{(3-len(str(sb)))*' '}{sb}  {bond[2]}  {0}  {0}  {0}  {0}\n")
                bstartindex += len(molecule.atoms)
            # end
            f.write("M  END\n")
            f.write("\n")